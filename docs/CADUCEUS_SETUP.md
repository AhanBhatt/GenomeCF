# Caduceus Setup

This document describes the environment used for the completed Caduceus-Ph runs in GenomeCF.

## Why a Separate Setup Is Needed

The completed Caduceus checkpoint depends on `mamba_ssm`, which did not run in the default Windows CPU environment used for the lighter baselines and paper-generation path. GenomeCF therefore treats Caduceus as:

- completed in WSL2/Linux CUDA
- skipped in the default Windows CPU environment

That keeps the benchmark runnable on CPU while still allowing the reverse-complement-aware baseline to be completed honestly.

## Completed Environment

- WSL2 Ubuntu
- NVIDIA GPU visible through WSL
- Python 3.10
- `torch 2.12.0+cu130`
- `transformers 4.38.1`
- `tokenizers 0.15.2`
- `mamba_ssm`
- `cuda-nvcc`

## Base Environment File

The base conda environment file is:

- `envs/caduceus.yml`

It creates the Python and CUDA compiler environment. `mamba_ssm` is installed afterward with `pip` because it needs the active CUDA compiler toolchain.

## Recommended Steps

From WSL:

```bash
conda env create -f envs/caduceus.yml
conda activate caduceus
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130
pip install transformers==4.38.1 tokenizers==0.15.2
pip install --no-build-isolation mamba-ssm
pip install -e .[benchmark,dev]
```

## Smoke Test

This should load Caduceus, tokenize two short sequences, and produce a tiny result row:

```bash
python -m pytest -m gpu tests/test_foundation_release.py -k caduceus
```

## One Benchmark Run

```bash
python -m genomecf.cli evaluate \
  --task human_nontata_promoters \
  --model caduceus_ph \
  --split official \
  --mode frozen \
  --seed 2026
```

## Notes

- Completed Caduceus rows in the release registry were generated through this WSL2/Linux CUDA route.
- The default Windows CPU environment still skips Caduceus cleanly.
- If `mamba_ssm` fails to build, verify that:
  - `torch.cuda.is_available()` is `True`
  - `nvcc --version` works inside the environment
  - `transformers==4.38.1` is installed
