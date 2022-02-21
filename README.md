# EAST

## Installation

### 1. Clone this repository

```bash
git clone https://github.com/4620511/EAST
```

### 2. Install dependencies

```bash
poetry install
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
