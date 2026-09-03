"""Trainable code encoder for the re-encode-in-loop GNN path.

The base pipeline freezes CodeBERT and mean-pools once, so the obfuscated
payload is averaged away before the GNN ever sees it. This module puts the
encoder back in the training graph — but only the top-N transformer layers are
trainable (the rest stay frozen) and, by default, we only re-encode the *live*
nodes (added/modified code), keeping compute bounded on graphs with hundreds of
unchanged context nodes.

No `peft` dependency: "finetune" == unfreeze the top-N encoder layers. Swap in
LoRA here later if desired.
"""

from __future__ import annotations

from typing import Any


def build_finetune_encoder(model_name: str, trainable_layers: int = 2):  # pragma: no cover - needs torch
    try:
        import torch
        from torch import nn
        from transformers import AutoModel
    except ImportError as exc:
        raise ImportError("Encoder finetune requires torch + transformers.") from exc

    class _Encoder(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.model = AutoModel.from_pretrained(model_name, use_safetensors=True)
            self.hidden_size = int(getattr(self.model.config, "hidden_size", 768))
            self.trainable_layers = int(trainable_layers)
            self._set_trainable()

        def _encoder_layers(self):
            # RoBERTa/BERT: model.encoder.layer is the ModuleList of blocks.
            enc = getattr(self.model, "encoder", None)
            return getattr(enc, "layer", None) if enc is not None else None

        def _set_trainable(self) -> None:
            for param in self.model.parameters():
                param.requires_grad = False
            layers = self._encoder_layers()
            if layers is None or self.trainable_layers <= 0:
                return
            for layer in layers[-self.trainable_layers:]:
                for param in layer.parameters():
                    param.requires_grad = True

        def forward(self, input_ids, attention_mask):
            if input_ids is None or input_ids.numel() == 0:
                device = next(self.parameters()).device
                return torch.zeros((0, self.hidden_size), device=device)
            out = self.model(input_ids=input_ids, attention_mask=attention_mask)
            hidden = out.last_hidden_state  # (n, L, H)
            mask = attention_mask.unsqueeze(-1).to(hidden.dtype)  # (n, L, 1)
            summed = (hidden * mask).sum(dim=1)
            counts = mask.sum(dim=1).clamp(min=1.0)
            return summed / counts

    return _Encoder()


def has_trainable_params(module: Any) -> bool:  # pragma: no cover - needs torch
    return any(p.requires_grad for p in module.parameters())
