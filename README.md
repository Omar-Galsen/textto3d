---
title: Ninja3D Studio
emoji: 🥷
colorFrom: gray
colorTo: green
sdk: gradio
sdk_version: 5.50.0
app_file: app.py
pinned: false
---
# Ninja3D Studio

A free, open-source improvement to the Idea2Mesh ninja workflow. It calls the public
[hysts/Shap-E Space](https://huggingface.co/spaces/hysts/Shap-E), generates up to four
seed variations sequentially, previews them side by side, cleans triangle meshes,
and exports GLB / OBJ / STL plus an auditable ZIP manifest.

## What improves

- Upload one of the four realistic ninja references instead of fighting the embedded upload control.
- Run 1–4 deterministic seed variations with the same settings.
- Inspect four actual mesh previews before choosing a design.
- Preserve raw output, remove duplicate/degenerate faces, ground and resize to 1.8 m.
- Download the full batch with seeds, settings, mesh counts and failures.
- Stop immediately on a failure instead of hammering a shared GPU quota.

**This is an inference workflow, not a trained new AI.** More seeds do not teach
Shap-E. Realistic faces, hands, clothing, anatomy and animation-ready topology are
not guaranteed. Use a stronger image-to-3D backend in a later revision if needed.

## Run on Windows / Git Bash

```bash
git clone https://github.com/Omar-Galsen/textto3d.git
cd textto3d
python -m venv .venv
source .venv/Scripts/activate
python -m pip install -r requirements.txt
python app.py
```

Open http://127.0.0.1:7860. Choose a reference or describe your ninja, start with
one variation, confirm the external upload, then generate.

## Android / online

Android uses the browser UI; it does not run GPU inference locally. For an online
version, create a free CPU Gradio Space on Hugging Face and upload these sources
(or sync this repository). The CPU Space hosts the UI; generation is performed
by the upstream shared GPU Space. GPU availability and free quotas are controlled
by Hugging Face. A free account/token may be required; set `HF_TOKEN` as a host
secret, never commit it. No paid API is used, but unlimited free generation is
not promised. If the upstream API rejects requests, use its direct demo or wait
for quota reset. This repository alone does not deploy a live service.

To use your phone on the same trusted Wi-Fi as your PC:

```bash
HOST=0.0.0.0 python app.py
```

Visit your PC's local IP on port 7860. This mode has no application authentication;
use only on a trusted local network, and do not port-forward it to the internet.

## Verification

```bash
python -m unittest discover -s tests -v
```

Tests cover independent seeds, all exports, normalization, ZIP manifests, input
validation and stopping on provider failures. Live shared-GPU generation and
visual character quality require separate testing. No automated quality score
claims to measure how good a ninja looks.

## Credits and next steps

Generation: [OpenAI Shap-E](https://github.com/openai/shap-e), served by
[hysts](https://huggingface.co/hysts). New reference images were created for this
project. Upstream licenses and service terms remain applicable.

Next: validate image-to-3D on the hosted API, add a stronger open-source backend,
then rig the chosen mesh and export FBX with Blender. Training, rigging, reference
image generation and FBX conversion are not implemented here.
