# PARITY

**Forecasts reveal their own errors through redundant path coding**

## Repository structure

```text
PARITY/
├── data_provider/
├── exp/
│   ├── exp_basic.py
│   └── exp_long_term_forecasting_parity.py
├── models/
│   └── parity.py
├── scripts/PARITY/
│   └── traffic.sh
├── dataset/README.md
├── run_parity.py
├── requirements.txt
├── VERSION
└── LICENSE
```

## Installation

```bash
conda create -n parity python=3.11
conda activate parity
pip install -r requirements.txt
```

## Dataset preparation

Place the Traffic file at:

```text
dataset/traffic/traffic.csv
```

See `dataset/README.md` for the public source.

## Run Traffic forecasting

Run all four horizons:

```bash
bash scripts/PARITY/traffic.sh
```

Run selected horizons:

```bash
bash scripts/PARITY/traffic.sh 96
bash scripts/PARITY/traffic.sh 192 336 720
```

## Acknowledgment

This implementation is built on
[Time-Series-Library](https://github.com/thuml/Time-Series-Library).

The upstream MIT license is retained.

