# Notebooks

Interactive visualizations for token-level watermark analysis.

## Notebooks

| Notebook | Description |
|---|---|
| [`notebook_watermark.ipynb`](notebook_watermark.ipynb) | **Token-Level Watermark Visualization** — reproduces the `cli_watermark.py` workflow. Each token is colored **green** if it matches the watermark key (green-list) or left uncolored if it doesn't (red-list). |
| [`notebook_watermark_synthid.ipynb`](notebook_watermark_synthid.ipynb) | **SynthID-Text Tournament Watermark Visualization** — reproduces `cli_watermark_synthid.py`. Tokens are colored by their **hit strength** (how many of the `m` watermarking layers score the token as "green") using a continuous viridis colormap: dark purple (weak signal) → bright yellow (strong signal). Colors are normalized across all three texts for direct comparison. |

## Re-executing

To regenerate both notebooks (execute all cells and save in-place):

```bash
uv run jupyter nbconvert --to notebook --execute --inplace notebooks/notebook_watermark_synthid.ipynb
uv run jupyter nbconvert --to notebook --execute --inplace notebooks/notebook_watermark.ipynb
```