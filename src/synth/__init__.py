"""LLM-based training-data synthesis for the structural misalignment GNN.

The featurebench obfuscated attack sets are monotonous: every FCV positive uses
one fixed base64 CWE-78 template, so the GNN memorizes the template instead of
learning "injection-ness." This package synthesizes *diverse* malicious (and
hard-negative benign) counterfactuals by injecting into real train-split
patches, keeping the real repo so the CFG/graph still builds.

Guardrails baked in (see scripts/synth_training_data.py):
  - draws only from an explicit train-instance allowlist (no heldout leakage)
  - validates every synthetic row (diff applies cleanly + payload present)
  - writes a separate jsonl wired into train_malicious paths only
"""
