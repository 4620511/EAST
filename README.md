# EAST

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
