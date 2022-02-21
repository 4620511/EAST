# EAST

## Requirements

- Python 3.9

## Installation

### 1. Clone this repository

```bash
git clone https://github.com/4620511/EAST
```

### 2. Install dependencies

```bash
poetry install
```

If you don't have `poetry`, get by running following command.

```bash
curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py | python -
```

In addition, install PyTorch that works in your environment.

Normally, adding by following command.

```bash
poetry add torch torchvision

```

Or, if your GPU is NVIDIA RTX A6000 or something like "sm_86" architecture, you have to adding by following command.

```bash
poetry run poe cuda113
```

### 3. Download pre-trained weights

```bash
./scripts/download.sh
```

## Usage

### Demo

```bash
poetry run poe demo
```

Default port is 8888.  
Change by editing `.streamlit/config.toml`.

### Use your dataset

```bash
poetry run python detect.py ./path/to/your/dataset --pattern "*.jpg"
```

More options available with `poetry run python detect.py -h`.
