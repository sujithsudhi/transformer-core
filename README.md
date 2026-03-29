# transformer-core

Reusable transformer building blocks extracted from application repositories.

## Package Layout

```text
transformer-core/
  pyproject.toml
  README.md
  src/
    transformer_core/
      __init__.py
      embeddings.py
      layers.py
  tests/
```

## Install

```powershell
python -m pip install -e .
```

## Release

Tag releases and let downstream repos pin to a Git tag:

```powershell
git tag v0.1.0
git push origin main --tags
```
