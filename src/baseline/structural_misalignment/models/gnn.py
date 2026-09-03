"""PyG heterogeneous GNN model for graph-level injection detection."""

from __future__ import annotations

from typing import Any, Dict


def _require_pyg():
    try:
        import torch
        import torch.nn.functional as F
        from torch import nn
        from torch_geometric.nn import GraphConv, HeteroConv, SAGEConv, global_mean_pool
    except ImportError as exc:  # pragma: no cover - dependency guard
        raise ImportError("Structural misalignment GNN requires torch-geometric.") from exc
    return torch, nn, F, GraphConv, HeteroConv, SAGEConv, global_mean_pool


_GROUNDS = ("subtask", "grounds", "code")
_DEPENDS = ("subtask", "depends_on", "subtask")
_CFG = ("code", "cfg", "code")


def model_options_from_metadata(metadata: Dict[str, Any]) -> Dict[str, Any]:
    """Reconstruct the model-architecture kwargs from a saved metadata dict, so
    load-time and inference-time models match the trained checkpoint exactly."""
    encoder = None
    if metadata.get("encoder_finetune"):
        encoder = {
            "model_name": str(metadata.get("embedding_model_name", "microsoft/codebert-base")),
            "trainable_layers": int(metadata.get("encoder_trainable_layers", 2)),
        }
    return {
        "input_dim": int(metadata.get("input_dim", 768)),
        "hidden_dim": int(metadata.get("hidden_dim", 128)),
        "dropout": float(metadata.get("dropout", 0.1)),
        "classifier_head": str(metadata.get("classifier_head", "mlp")),
        "residual_dim": int(metadata.get("residual_dim", 0)),
        "aux_node_head": bool(metadata.get("aux_node_head", False)),
        "edge_weighted": bool(metadata.get("edge_weighted_grounding", False)),
        "encoder": encoder,
    }


class HeteroGraphClassifier:  # pragma: no cover - thin wrapper around torch module
    def __new__(cls, *args, **kwargs):
        torch, nn, F, GraphConv, HeteroConv, SAGEConv, global_mean_pool = _require_pyg()

        def _conv_block(in_dim, out_dim, edge_weighted):
            relations = {
                _DEPENDS: SAGEConv((in_dim, in_dim), out_dim),
                _CFG: SAGEConv((in_dim, in_dim), out_dim),
                # GraphConv consumes an edge_weight; SAGEConv ignores it.
                _GROUNDS: (GraphConv((in_dim, in_dim), out_dim) if edge_weighted
                           else SAGEConv((in_dim, in_dim), out_dim)),
            }
            return HeteroConv(relations, aggr="sum")

        class _Model(nn.Module):
            def __init__(
                self,
                input_dim: int = 768,
                hidden_dim: int = 128,
                dropout: float = 0.1,
                classifier_head: str = "mlp",
                residual_dim: int = 0,
                aux_node_head: bool = False,
                edge_weighted: bool = False,
                encoder: Dict[str, Any] | None = None,
            ) -> None:
                super().__init__()
                self.edge_weighted = bool(edge_weighted)
                self.conv1 = _conv_block(input_dim, hidden_dim, self.edge_weighted)
                self.conv2 = _conv_block(hidden_dim, hidden_dim, self.edge_weighted)
                self.dropout = float(dropout)
                self.hidden_dim = int(hidden_dim)
                self.residual_dim = int(residual_dim)

                self.encoder = None
                self.encoder_dim = 0
                if encoder is not None:
                    from src.baseline.structural_misalignment.models.encoder_head import (
                        build_finetune_encoder,
                    )

                    self.encoder = build_finetune_encoder(
                        str(encoder.get("model_name", "microsoft/codebert-base")),
                        int(encoder.get("trainable_layers", 2)),
                    )
                    self.encoder_dim = int(self.encoder.hidden_size)

                readout_dim = hidden_dim * 2 + self.residual_dim
                if classifier_head == "logistic":
                    self.classifier = nn.Linear(readout_dim, 2)
                else:
                    self.classifier = nn.Sequential(
                        nn.Linear(readout_dim, hidden_dim),
                        nn.ReLU(),
                        nn.Dropout(self.dropout),
                        nn.Linear(hidden_dim, 2),
                    )
                self.aux_head = nn.Linear(hidden_dim, 1) if aux_node_head else None

            # -- encoder re-encode of live nodes ---------------------------------
            def _apply_encoder(self, store, x):
                if self.encoder is None or x is None or x.numel() == 0:
                    return x
                ids = getattr(store, "input_ids", None)
                mask = getattr(store, "attention_mask", None)
                live = getattr(store, "live_mask", None)
                if ids is None or mask is None or live is None or not bool(live.any()):
                    return x
                idx = live.nonzero(as_tuple=True)[0]
                enc = self.encoder(ids[idx], mask[idx]).to(x.dtype)
                base = x[:, : self.encoder_dim].clone().index_copy(0, idx, enc)
                extra = x[:, self.encoder_dim:]
                return torch.cat([base, extra], dim=-1) if extra.size(1) else base

            def _node_inputs(self, data):
                x_dict = {key: value for key, value in data.x_dict.items()}
                if self.encoder is not None:
                    for node_type in ("code", "subtask"):
                        if node_type in x_dict:
                            x_dict[node_type] = self._apply_encoder(data[node_type], x_dict[node_type])
                return x_dict

            def _grounds_weight(self, data):
                if not self.edge_weighted:
                    return {}
                store = data[_GROUNDS]
                attr = getattr(store, "edge_attr", None)
                if attr is None:
                    return {}
                return {_GROUNDS: attr.view(-1)}

            def _convs(self, x_dict, data):
                ew = self._grounds_weight(data)
                edge_index_dict = data.edge_index_dict
                if ew:
                    x_dict = self.conv1(x_dict, edge_index_dict, edge_weight_dict=ew)
                    x_dict = {key: F.relu(value) for key, value in x_dict.items()}
                    x_dict = self.conv2(x_dict, edge_index_dict, edge_weight_dict=ew)
                else:
                    x_dict = self.conv1(x_dict, edge_index_dict)
                    x_dict = {key: F.relu(value) for key, value in x_dict.items()}
                    x_dict = self.conv2(x_dict, edge_index_dict)
                x_dict = {key: F.relu(value) for key, value in x_dict.items()}
                return x_dict

            def _pool(self, x_dict, batch_dict, num_graphs):
                pooled = []
                hidden_dim = self.hidden_dim
                device = next(self.parameters()).device
                for node_type in ("subtask", "code"):
                    x = x_dict.get(node_type)
                    if x is None or x.numel() == 0:
                        pooled.append(torch.zeros((num_graphs, hidden_dim), device=device))
                        continue
                    if node_type in batch_dict:
                        pooled.append(global_mean_pool(x, batch_dict[node_type], size=num_graphs))
                    else:
                        pooled.append(x.mean(dim=0, keepdim=True))
                return torch.cat(pooled, dim=-1)

            def run(self, data):
                """Full pass returning graph logits, per-code-node aux logits, and
                the pooled graph embedding (for contrastive training)."""
                x_dict = self._node_inputs(data)
                x_dict = self._convs(x_dict, data)
                try:
                    batch_dict = data.batch_dict
                except KeyError:
                    batch_dict = {}
                num_graphs = int(data.y.view(-1).size(0)) if hasattr(data, "y") else 1
                pooled = self._pool(x_dict, batch_dict, num_graphs)
                embedding = pooled
                readout = pooled
                if self.residual_dim > 0:
                    residual = getattr(data, "grounding_residual", None)
                    if residual is None:
                        residual = torch.zeros((num_graphs, self.residual_dim), device=pooled.device)
                    else:
                        residual = residual.view(num_graphs, self.residual_dim).to(pooled.device)
                    readout = torch.cat([pooled, residual], dim=-1)
                logits = self.classifier(readout)
                node_logits = None
                if self.aux_head is not None:
                    code_x = x_dict.get("code")
                    if code_x is not None and code_x.numel() > 0:
                        node_logits = self.aux_head(code_x).view(-1)
                return {"logits": logits, "node_logits": node_logits, "embedding": embedding}

            def forward(self, data):
                return self.run(data)["logits"]

        return _Model(*args, **kwargs)
