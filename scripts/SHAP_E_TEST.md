# Test the retained original Shap-E code

The unmodified Python source and notebooks are copied under `vendor/shap-e`.
`UPSTREAM.json` identifies upstream commit and SHA-256 checksums; `LICENSE`
retains OpenAI's MIT license. Large sample GIFs and example mesh assets were
excluded; notebooks referring to example_data need those assets from upstream.
No model weights or trained new model are stored in this repository.

## Windows / Git Bash

Use Python 3.10 or 3.11 for upstream compatibility. From the repository root:

```bash
python -m venv .venv
source .venv/Scripts/activate
python -m pip install --upgrade pip
# Install the CUDA-enabled PyTorch build appropriate for your GPU first:
# https://pytorch.org/get-started/locally/
python -m pip install -e vendor/shap-e
python -m pip install ipywidgets
python scripts/test_shap_e.py --verify-source
python scripts/test_shap_e.py --prompt "A full body ninja wearing black clothing" --count 1
```

First generation downloads the original pretrained transmitter and text300M
weights. It runs on your own GPU (or slowly on CPU), independent of Meshy and
the public Shap-E demo. There is no paid API. Internet is needed for initial
installation and weight downloads. Open the OBJ in Blender.

```bash
python scripts/test_shap_e.py --count 4 --seed 10 --output outputs/ninja-batch
```

Each variation runs sequentially to reduce peak memory. Seed/settings are saved
in manifest.json. More variations do not train Shap-E. Character anatomy and
realism remain limited by the original pretrained model. This is a CLI test;
the existing Gradio app still uses the hosted Space adapter.

The source checksum check and Python compilation can run without weights.
Full neural inference was not executed in the authoring environment.
