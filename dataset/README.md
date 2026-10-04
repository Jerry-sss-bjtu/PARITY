# Data setup

PARITY uses seven public multivariate forecasting datasets. The raw data are **not redistributed with this repository**; please obtain them from the original public sources and place the CSV files in the directory structure shown below.

## Expected directory layout

```text
dataset/
├── ETT-small/
│   ├── ETTh1.csv
│   ├── ETTh2.csv
│   ├── ETTm1.csv
│   └── ETTm2.csv
├── weather/
│   └── weather.csv
├── electricity/
│   └── electricity.csv
└── traffic/
    └── traffic.csv
```

## Benchmark files

| Benchmark | Channels | Temporal resolution | Local path | Public source |
|:--|--:|:--|:--|:--|
| ETTh1 | 7 | 1 hour | `dataset/ETT-small/ETTh1.csv` | [ETT-small](https://www.kaggle.com/datasets/alaaelmor/ettsmall) |
| ETTh2 | 7 | 1 hour | `dataset/ETT-small/ETTh2.csv` | [ETT-small](https://www.kaggle.com/datasets/alaaelmor/ettsmall) |
| ETTm1 | 7 | 15 minutes | `dataset/ETT-small/ETTm1.csv` | [ETT-small](https://www.kaggle.com/datasets/alaaelmor/ettsmall) |
| ETTm2 | 7 | 15 minutes | `dataset/ETT-small/ETTm2.csv` | [ETT-small](https://www.kaggle.com/datasets/alaaelmor/ettsmall) |
| Weather | 21 | 10 minutes | `dataset/weather/weather.csv` | [Jena weather data](https://www.bgc-jena.mpg.de/wetter/weather_data.html) |
| Electricity | 321 | 1 hour | `dataset/electricity/electricity.csv` | [Zenodo record 4656140](https://zenodo.org/records/4656140) |
| Traffic | 862 | 1 hour | `dataset/traffic/traffic.csv` | [Zenodo record 4656132](https://zenodo.org/records/4656132) |

## Preparation notes

### ETT family

Download the ETT-small package and copy the four benchmark files into:

```text
dataset/ETT-small/
```

The expected filenames are:

```text
ETTh1.csv
ETTh2.csv
ETTm1.csv
ETTm2.csv
```

### Weather

Obtain the Weather data from the Jena climate-data archive and save the benchmark CSV as:

```text
dataset/weather/weather.csv
```

### Electricity

Download the Electricity benchmark from its Zenodo record and place it at:

```text
dataset/electricity/electricity.csv
```

### Traffic

Download the Traffic benchmark from its Zenodo record and place it at:

```text
dataset/traffic/traffic.csv
```

## Before running experiments

Check that all required CSV files exist at the paths above. The experiment scripts assume this directory organization and do not download datasets automatically.
