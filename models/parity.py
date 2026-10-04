
from __future__ import annotations

from typing import Optional, Tuple, Dict
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _ensure_mark4(x_mark: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    if x_mark is None:
        return None
    if x_mark.shape[-1] == 4:
        return x_mark
    if x_mark.shape[-1] > 4:
        return x_mark[..., :4].contiguous()
    pad = torch.zeros(
        *x_mark.shape[:-1],
        4 - x_mark.shape[-1],
        dtype=x_mark.dtype,
        device=x_mark.device,
    )
    return torch.cat([x_mark, pad], dim=-1)


def destationary_norm(
    x: torch.Tensor,
    eps: float = 1e-5
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    mean = x.mean(dim=1, keepdim=True).detach()
    var = x.var(dim=1, keepdim=True, unbiased=False).detach()
    std = torch.sqrt(var + eps)
    return (x - mean) / std, mean, std


def destationary_denorm(
    x: torch.Tensor,
    mean: torch.Tensor,
    std: torch.Tensor
) -> torch.Tensor:
    return x * std + mean


def build_path_parity_codebook(
    horizon: int,
    n_checks: int
) -> torch.Tensor:


    H = int(horizon)
    K = max(1, min(int(n_checks), H))
    t = torch.arange(H, dtype=torch.float64)

    rows = []
    n_dct = min(max(4, K // 2), K)
    for k in range(n_dct):
        if k == 0:
            row = torch.ones(H, dtype=torch.float64)
        else:
            row = torch.cos(math.pi * (t + 0.5) * k / H)
        row = row / row.norm().clamp_min(1e-12)
        rows.append(row)


    level = 0
    while len(rows) < K:
        blocks = 2 ** level
        block_len = H / blocks
        for b in range(blocks):
            if len(rows) >= K:
                break
            left = int(round(b * block_len))
            right = int(round((b + 1) * block_len))
            mid = (left + right) // 2
            if right - left < 2:
                continue
            row = torch.zeros(H, dtype=torch.float64)
            row[left:mid] = 1.0
            row[mid:right] = -1.0
            row = row / row.norm().clamp_min(1e-12)
            rows.append(row)
        level += 1
        if level > 12:
            break


    while len(rows) < K:
        k = len(rows)
        row = torch.cos(math.pi * (t + 0.5) * k / H)
        row = row / row.norm().clamp_min(1e-12)
        rows.append(row)

    candidate = torch.stack(rows[:K], dim=0)

    q, r = torch.linalg.qr(candidate.transpose(0, 1), mode="reduced")
    sign = torch.sign(torch.diag(r))
    sign = torch.where(sign == 0, torch.ones_like(sign), sign)
    q = q * sign.unsqueeze(0)
    codebook = q.transpose(0, 1).contiguous().float()
    return codebook


class ContextCoordinateEncoder(nn.Module):
    def __init__(self, d_model: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x_mark: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if x_mark is None:
            return None
        x_mark = _ensure_mark4(x_mark).float()
        return self.norm(self.net(x_mark))


class HistoryEncoder(nn.Module):


    def __init__(
        self,
        seq_len: int,
        c_in: int,
        d_model: int,
        dropout: float
    ):
        super().__init__()
        self.seq_len = int(seq_len)
        self.c_in = int(c_in)
        self.d_model = int(d_model)

        self.time_proj = nn.Linear(self.seq_len, d_model)
        self.seq_proj = nn.Linear(self.seq_len, d_model)
        self.trend_proj = nn.Linear(self.c_in, d_model)

        self.out = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
        )

    def forward(self, x_norm: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, L, C = x_norm.shape
        xc = x_norm.transpose(1, 2).contiguous()
        ch_tokens = self.time_proj(xc)
        ch_state = ch_tokens.mean(dim=1)

        global_seq = x_norm.mean(dim=-1)
        seq_state = self.seq_proj(global_seq)

        recent_k = min(4, L)
        trend = x_norm[:, -1, :] - x_norm[:, -recent_k:, :].mean(dim=1)
        trend_state = self.trend_proj(trend)

        hist_state = self.out(ch_state + seq_state + trend_state)
        return ch_tokens, hist_state


class FutureFrameField(nn.Module):


    def __init__(
        self,
        pred_len: int,
        d_model: int,
        dropout: float
    ):
        super().__init__()
        self.pred_len = int(pred_len)
        self.d_model = int(d_model)

        self.horizon_frame = nn.Parameter(
            torch.randn(1, self.pred_len, d_model) * 0.02
        )
        self.history_lift = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
        )
        self.frame_connection = nn.Sequential(
            nn.Conv1d(d_model, d_model, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(d_model, d_model, kernel_size=3, padding=1),
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self,
        hist_state: torch.Tensor,
        context_emb: Optional[torch.Tensor]
    ) -> torch.Tensor:
        B = hist_state.shape[0]
        frame = self.horizon_frame.expand(B, -1, -1)
        frame = frame + self.history_lift(hist_state).unsqueeze(1)
        if context_emb is not None:
            frame = frame + context_emb
        frame = frame + self.frame_connection(
            frame.transpose(1, 2)
        ).transpose(1, 2)
        return self.norm(frame)


class HistoryPathIntegral(nn.Module):


    def __init__(self, seq_len: int, pred_len: int):
        super().__init__()
        self.seq_len = int(seq_len)
        self.pred_len = int(pred_len)

        self.kernel = nn.Parameter(
            torch.empty(self.pred_len, self.seq_len)
        )
        self.boundary = nn.Parameter(torch.empty(self.pred_len))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.kernel, a=math.sqrt(5))
        fan_in = self.seq_len
        bound = 1.0 / math.sqrt(fan_in) if fan_in > 0 else 0.0
        nn.init.uniform_(self.boundary, -bound, bound)

    def forward(self, x_norm: torch.Tensor) -> torch.Tensor:
        anchor = torch.einsum(
            "hl,blc->bhc",
            self.kernel,
            x_norm
        )
        return anchor + self.boundary.view(1, -1, 1)


class TrajectoryFrameTransport(nn.Module):


    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        c_out: int,
        d_model: int,
        rank: int = 8
    ):
        super().__init__()
        self.seq_len = int(seq_len)
        self.pred_len = int(pred_len)
        self.c_out = int(c_out)
        self.rank = int(rank)

        self.anchor_propagator = HistoryPathIntegral(
            self.seq_len,
            self.pred_len
        )


        self.chart_head = nn.Linear(d_model, 2 * self.rank)
        self.dilation_modes = nn.Parameter(
            torch.randn(self.rank, self.c_out) * 0.02
        )
        self.drift_modes = nn.Parameter(
            torch.randn(self.rank, self.c_out) * 0.02
        )


        self.dilation_bound = nn.Parameter(torch.tensor(0.10))
        self.drift_scale = nn.Parameter(torch.tensor(0.10))

    def forward(
        self,
        x_norm: torch.Tensor,
        frame_state: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        anchor_path = self.anchor_propagator(x_norm)

        chart_coef = self.chart_head(frame_state)
        dilation_coef, drift_coef = chart_coef.chunk(2, dim=-1)

        dilation_coordinate = torch.einsum(
            "bhr,rc->bhc",
            dilation_coef,
            self.dilation_modes
        )
        drift_coordinate = torch.einsum(
            "bhr,rc->bhc",
            drift_coef,
            self.drift_modes
        )

        local_jacobian = 1.0 + self.dilation_bound * torch.tanh(
            dilation_coordinate
        )
        tangent_displacement = self.drift_scale * drift_coordinate

        transported_path = (
            local_jacobian * anchor_path
            + tangent_displacement
        )

        return {
            "y_hat": transported_path,
            "anchor_path": anchor_path,
            "local_jacobian": local_jacobian,
            "tangent_displacement": tangent_displacement,
            "dilation_coordinate": dilation_coordinate,
            "drift_coordinate": drift_coordinate,
        }


class PathParityPredictor(nn.Module):


    def __init__(
        self,
        n_checks: int,
        d_model: int,
        dropout: float
    ):
        super().__init__()
        self.n_checks = int(n_checks)
        self.d_model = int(d_model)

        self.check_pos = nn.Parameter(
            torch.randn(1, self.n_checks, d_model) * 0.02
        )
        self.hist_proj = nn.Linear(d_model, d_model)
        self.mark_proj = nn.Linear(d_model, d_model)
        self.channel_proj = nn.Linear(d_model, d_model)
        self.channel_to_checks = nn.Linear(d_model, self.n_checks)
        self.check_residual_scale = nn.Parameter(torch.tensor(-1.5))

        self.code_norm = nn.LayerNorm(d_model)
        self.channel_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self.reliability = nn.Linear(d_model, 1)
        nn.init.zeros_(self.reliability.weight)
        nn.init.constant_(self.reliability.bias, -2.0)

    def forward(
        self,
        ch_tokens: torch.Tensor,
        hist_state: torch.Tensor,
        context_emb: Optional[torch.Tensor],
        abs_codebook: torch.Tensor,
        base_checks: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, C, D = ch_tokens.shape
        K, H = abs_codebook.shape

        code_state = self.check_pos.expand(B, -1, -1)
        code_state = code_state + self.hist_proj(hist_state).unsqueeze(1)

        if context_emb is not None:
            denom = abs_codebook.sum(dim=-1, keepdim=True).clamp_min(1e-6)
            mark_code = torch.einsum(
                "kh,bhd->bkd",
                abs_codebook,
                context_emb
            ) / denom.unsqueeze(0)
            code_state = code_state + self.mark_proj(mark_code)

        code_state = self.code_norm(code_state)
        channel_state = self.channel_norm(self.channel_proj(ch_tokens))

        interaction = torch.einsum(
            "bkd,bcd->bkc",
            self.dropout(code_state),
            self.dropout(channel_state)
        ) / math.sqrt(float(D))

        channel_bias = self.channel_to_checks(ch_tokens).transpose(1, 2)
        check_residual = torch.tanh(self.check_residual_scale) * 0.10 * (
            interaction + channel_bias
        )

        predicted_checks = base_checks + check_residual
        reliability = torch.sigmoid(self.reliability(code_state))
        return predicted_checks, reliability, check_residual


class PARITYForecaster(nn.Module):


    def __init__(self, args):
        super().__init__()

        self.seq_len = int(args.seq_len)
        self.label_len = int(getattr(args, "label_len", 0))
        self.pred_len = int(args.pred_len)
        self.enc_in = int(args.enc_in)
        self.c_out = int(getattr(args, "c_out", args.enc_in))
        self.d_model = int(getattr(args, "d_model", 64))
        self.dropout = float(getattr(args, "dropout", 0.1))

        self.rank = int(getattr(args, "path_rank", 8))
        requested_checks = int(getattr(args, "parity_checks", 16))
        self.n_checks = max(1, min(requested_checks, self.pred_len))


        codebook = build_path_parity_codebook(
            horizon=self.pred_len,
            n_checks=self.n_checks
        )
        self.register_buffer("codebook", codebook, persistent=True)
        self.register_buffer(
            "abs_codebook",
            codebook.abs(),
            persistent=False
        )

        self.sqrt_h = math.sqrt(float(self.pred_len))

        self.context_encoder = ContextCoordinateEncoder(self.d_model, self.dropout)
        self.history_encoder = HistoryEncoder(
            self.seq_len,
            self.enc_in,
            self.d_model,
            self.dropout
        )
        self.frame_field = FutureFrameField(
            self.pred_len,
            self.d_model,
            self.dropout
        )
        self.path_transport = TrajectoryFrameTransport(
            self.seq_len,
            self.pred_len,
            self.c_out,
            self.d_model,
            rank=self.rank
        )
        self.check_predictor = PathParityPredictor(
            self.n_checks,
            self.d_model,
            self.dropout
        )


        self.decode_gain_logit = nn.Parameter(torch.tensor(-2.0))

    def _encode_checks(self, y_norm: torch.Tensor) -> torch.Tensor:
        return torch.einsum(
            "kh,bhc->bkc",
            self.codebook,
            y_norm
        ) / self.sqrt_h

    def _decode_syndrome(
        self,
        syndrome: torch.Tensor
    ) -> torch.Tensor:
        return self.sqrt_h * torch.einsum(
            "hk,bkc->bhc",
            self.codebook.transpose(0, 1),
            syndrome
        )

    def _make_output(
        self,
        x_enc: torch.Tensor,
        y_hat: torch.Tensor
    ) -> torch.Tensor:
        out = torch.zeros(
            x_enc.shape[0],
            self.label_len + self.pred_len,
            self.c_out,
            device=x_enc.device,
            dtype=y_hat.dtype
        )
        out[:, -self.pred_len:, :] = y_hat
        return out

    def forward(
        self,
        x_enc,
        x_mark_enc,
        x_dec,
        x_mark_dec,
        y_true=None,
        epoch_idx_1based=None
    ):
        x_mark_dec = _ensure_mark4(x_mark_dec)
        context_coord = (
            None if x_mark_dec is None
            else x_mark_dec[:, -self.pred_len:, :]
        )
        context_emb = self.context_encoder(context_coord)

        x_norm, mean, std = destationary_norm(x_enc)
        ch_tokens, hist_state = self.history_encoder(x_norm)
        frame_state = self.frame_field(hist_state, context_emb)

        dec = self.path_transport(x_norm, frame_state)

        point_hat_n = dec["y_hat"]
        point_hat = destationary_denorm(point_hat_n, mean, std)
        base_checks = self._encode_checks(point_hat_n)

        predicted_checks, reliability, check_residual = self.check_predictor(
            ch_tokens=ch_tokens,
            hist_state=hist_state,
            context_emb=context_emb,
            abs_codebook=self.abs_codebook,
            base_checks=base_checks
        )

        syndrome = predicted_checks - base_checks
        decode_gain = torch.sigmoid(self.decode_gain_logit)

        weighted_syndrome = reliability * syndrome
        correction_n = decode_gain * self._decode_syndrome(
            weighted_syndrome
        )

        correction_n = 0.75 * torch.tanh(correction_n / 0.75)
        final_hat_n = point_hat_n + correction_n

        final_hat = destationary_denorm(final_hat_n, mean, std)
        out = self._make_output(x_enc, final_hat)


        if y_true is None:
            return out

        if (
            y_true.dim() == 3
            and y_true.shape[1] == self.label_len + self.pred_len
        ):
            y_fut = y_true[:, -self.pred_len:, :].contiguous()
        elif y_true.dim() == 3 and y_true.shape[1] == self.pred_len:
            y_fut = y_true
        else:
            raise ValueError(
                f"Unexpected y_true shape: {tuple(y_true.shape)}"
            )


        with torch.autocast(
            device_type=y_fut.device.type,
            enabled=False
        ):
            y_fut_n = (
                y_fut.float() - mean.float()
            ) / std.float()
            true_checks = self._encode_checks(y_fut_n)
        final_checks = self._encode_checks(final_hat_n)


        l_pred = F.smooth_l1_loss(final_hat, y_fut)
        l_point = F.smooth_l1_loss(point_hat, y_fut)


        with torch.autocast(
            device_type=predicted_checks.device.type,
            enabled=False
        ):
            l_check = F.smooth_l1_loss(
                predicted_checks.float(),
                true_checks.float()
            )
        l_consistency = F.mse_loss(
            final_checks,
            predicted_checks.detach()
        )


        l_total = (
            l_pred
            + 0.10 * l_point
            + 0.05 * l_check
            + 0.01 * l_consistency
        )

        loss_dict = {
            "L_total": l_total,
            "L_pred": l_pred.detach(),
            "L_point": l_point.detach(),
            "L_check": l_check.detach(),
            "L_consistency": l_consistency.detach(),
            "syndrome_energy": syndrome.pow(2).mean().detach(),
            "correction_energy": correction_n.pow(2).mean().detach(),
            "mean_reliability": reliability.mean().detach(),
            "decode_gain": decode_gain.detach(),
        }


        return out, loss_dict


Model = PARITYForecaster
parity = PARITYForecaster

