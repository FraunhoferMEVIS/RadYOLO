# Development

The main source code lives in the `radyolo/` package and documentation in the `documentation/` folder.

## Run tests

End-to-end CLI tests live in `tests/`. Install dev dependencies and run:

```bash
uv run --group dev pytest
```

## Coverage analysis

```bash
uv run --group dev pytest --html=pytest-report.html --self-contained-html --cov=radyolo --cov-report=term --cov-report=xml --cov-report=html
```
