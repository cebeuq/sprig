---
title: SPRIG v0.1 Demo
emoji: 🔺
colorFrom: red
colorTo: blue
sdk: gradio
sdk_version: 5.9.1
python_version: "3.11"
app_file: app.py
pinned: false
license: mit
---

# SPRIG v0.1 — interactive demo

A text-to-image **scene grammar** where images are *derived*, not denoised.
Two tabs: **prompt → image** (derive a 64×64 sample) and **image → parse**
(infer an image's most likely grammar derivation and overlay its region tree —
SPRIG's strongest capability).

Research preview at 64×64. See the [model card](https://huggingface.co/cebeuq/sprig)
for the scorecard: object binding is partial, but exact
likelihood, caption information gain, and parsing all work.

Runs on CPU (~16M-param model). Set the `SPRIG_MODEL_REPO` space secret to the
model repo id.
