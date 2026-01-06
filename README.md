---
title: Chess Challenge Arena
emoji: ♟️
colorFrom: gray
colorTo: yellow
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: true
license: mit
---

# Chess Challenge Arena

This Space hosts the evaluation arena for the LLM Chess Challenge.

**Chess Challenge Template**: https://github.com/nathanael-fijalkow/ChessChallengeTemplate

## Features

- **Interactive Demo**: Test any submitted model against Stockfish
- **Leaderboard**: See rankings of all submitted models
- **Statistics**: View detailed performance metrics

## Setup (Admin)

### 1. Create a Private Leaderboard Dataset

Create a private dataset to store the leaderboard CSV:

```bash
# Using the HuggingFace CLI
huggingface-cli repo create chess-challenge-leaderboard --type dataset --private
```

Or create it via the web UI at: https://huggingface.co/new-dataset

### 2. Configure Space Secrets

Go to **Settings → Variables and secrets** and add:

| Secret/Variable | Value | Description |
|-----------------|-------|-------------|
| `HF_TOKEN` | `hf_xxx...` | Write-access token for the leaderboard dataset |
| `HF_ORGANIZATION` | `LLM-course` | Your organization name |
| `LEADERBOARD_DATASET` | `LLM-course/chess-challenge-leaderboard` | Dataset repo ID |

> ⚠️ The `HF_TOKEN` needs **write access** to the leaderboard dataset to save results.

## How to Submit

Students should push their trained models to this organization:

```python
from chess_challenge import ChessForCausalLM, ChessTokenizer

model.push_to_hub("your-model-name", organization="LLM-course")
tokenizer.push_to_hub("your-model-name", organization="LLM-course")
```

Models will be automatically evaluated and added to the leaderboard.
