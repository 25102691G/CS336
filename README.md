## Setup

### Prerequisites

1. Python 3.11+
2. Install `uv` [here](https://github.com/astral-sh/uv) (recommended) or run `pip install uv`/`brew install uv`

### Running Individual Assignments

Each assignment is self-contained. Navigate to the assignment directory and use `uv run`:

```bash
cd assignment1-basics
uv run pytest                    # Run tests
uv run python scripts/train.py   # Run training script
```

Dependencies are automatically installed based on each assignment's `pyproject.toml`.

## Assignments

### [Assignment 1: Basics](./assignment1-basics)

**What you will implement**
1. Byte-pair encoding (BPE) tokenizer
2. Transformer language model (LM)
3. The cross-entropy loss function and the AdamW optimizer
4. The training loop, with support for serializing and loading model and optimizer state

**What you will run**
1. Train a BPE tokenizer on the TinyStories dataset.
2. Run your trained tokenizer on the dataset to convert it into a sequence of integer IDs.
3. Train a Transformer LM on the TinyStories dataset.
4. Generate samples and evaluate perplexity using the trained Transformer LM.
5. Train models on OpenWebText and submit your attained perplexities to a leaderboard.
