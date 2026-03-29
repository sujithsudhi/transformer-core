# transformer-core

Reusable transformer building blocks extracted from application repositories.

The package currently provides:

- token embeddings
- sinusoidal or trainable positional encodings
- multi-head self-attention
- encoder and decoder transformer layers with optional KV caching

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

For development:

```powershell
python -m pip install -e .[dev]
```

## Quick Start

```python
import torch
from transformer_core import PositionalEncoding, TokenEmbedding, TransformerDecoderLayer

tokens = torch.tensor([[1, 2, 3, 4]])
embedding = TokenEmbedding(vocab_size=32, embed_dim=16)
positions = PositionalEncoding(max_len=32, embed_dim=16, dropout=0.0)
decoder = TransformerDecoderLayer(embedDim=16, numHeads=4)

x = positions(embedding(tokens))
y = decoder(x)
print(y.shape)  # torch.Size([1, 4, 16])
```

## Tests

```powershell
python -m pytest
```

## Release

Tag releases and let downstream repos pin to a Git tag:

```powershell
git tag v0.1.0
git push origin main --tags
```
