"""Training helpers for the structural misalignment hetero GNN."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from src.baseline.structural_misalignment.models.gnn import HeteroGraphClassifier


def _require_training_deps():
    try:
        import torch
        from torch.utils.data import WeightedRandomSampler
        from torch_geometric.loader import DataLoader
        from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
    except ImportError as exc:  # pragma: no cover - dependency guard
        raise ImportError(
            "Structural misalignment GNN training requires torch, torch-geometric, and scikit-learn."
        ) from exc
    return torch, WeightedRandomSampler, DataLoader, accuracy_score, precision_score, recall_score, roc_auc_score


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch, _, _, _, _, _, _ = _require_training_deps()
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _graph_label(graph) -> int:
    return int(graph.y.view(-1)[0].item())


def _infer_input_dim(graphs: List[Any], default: int = 768) -> int:
    """Node feature dimension (uniform across subtask/code by construction)."""
    for graph in graphs:
        for node_type in ("code", "subtask"):
            x = graph[node_type].x if node_type in graph.node_types else None
            if x is not None and x.dim() == 2 and x.size(1) > 0:
                return int(x.size(1))
    return default


def _infer_residual_dim(graphs: List[Any]) -> int:
    """Length of the grounding-residual readout vector, 0 if not attached."""
    for graph in graphs:
        residual = getattr(graph, "grounding_residual", None)
        if residual is not None:
            return int(residual.view(-1).size(0))
    return 0


def _model_flags(graph_feature_flags: Dict[str, Any] | None) -> Dict[str, Any]:
    """Translate the graph augmentation flags into model-architecture kwargs."""
    flags = graph_feature_flags or {}
    encoder = None
    if bool(flags.get("encoder_finetune", False)):
        encoder = {
            "model_name": str(flags.get("embedding_model_name", "microsoft/codebert-base")),
            "trainable_layers": int(flags.get("encoder_trainable_layers", 2)),
        }
    return {
        "aux_node_head": bool(flags.get("node_aux_labels", False)),
        "edge_weighted": bool(flags.get("edge_weighted_grounding", False)),
        "encoder": encoder,
    }


def _aux_node_loss(out, batch, torch):
    """Masked BCE on per-code-node injection labels (only synth-malicious graphs
    carry markers, so per-graph mask zeroes out everyone else)."""
    node_logits = out.get("node_logits")
    if node_logits is None:
        return None
    code_store = batch["code"]
    labels = getattr(code_store, "inj_label", None)
    if labels is None or labels.numel() == 0:
        return None
    import torch.nn.functional as F

    node_batch = getattr(code_store, "batch", None)
    graph_mask = getattr(batch, "node_label_mask", None)
    if node_batch is not None and graph_mask is not None:
        per_node_mask = graph_mask.view(-1)[node_batch].to(node_logits.dtype)
    else:
        per_node_mask = torch.ones_like(node_logits)
    bce = F.binary_cross_entropy_with_logits(node_logits, labels.to(node_logits.dtype), reduction="none")
    denom = per_node_mask.sum().clamp(min=1.0)
    return (bce * per_node_mask).sum() / denom


def _supervised_contrastive(embedding, labels, torch, temperature: float = 0.1):
    """SupCon over the batch: pull same-label graph embeddings together, push
    different-label apart. Returns None when a batch has <2 usable anchors."""
    import torch.nn.functional as F

    if embedding is None or embedding.size(0) < 2:
        return None
    labels = labels.view(-1)
    if len(torch.unique(labels)) < 2:
        return None
    z = F.normalize(embedding, dim=1)
    sim = torch.matmul(z, z.t()) / temperature
    sim = sim - sim.max(dim=1, keepdim=True).values.detach()
    n = z.size(0)
    logits_mask = 1.0 - torch.eye(n, device=z.device)
    pos_mask = (labels.unsqueeze(0) == labels.unsqueeze(1)).float() * logits_mask
    exp = torch.exp(sim) * logits_mask
    log_prob = sim - torch.log(exp.sum(dim=1, keepdim=True) + 1e-12)
    pos_counts = pos_mask.sum(dim=1)
    valid = pos_counts > 0
    if not bool(valid.any()):
        return None
    mean_log_prob_pos = (pos_mask * log_prob).sum(dim=1)[valid] / pos_counts[valid]
    return -mean_log_prob_pos.mean()


def _build_loader(graphs: List[Any], batch_size: int, weighted: bool):
    torch, WeightedRandomSampler, DataLoader, _, _, _, _ = _require_training_deps()
    if not graphs:
        return DataLoader([], batch_size=batch_size)
    if not weighted:
        return DataLoader(graphs, batch_size=batch_size, shuffle=False)
    labels = [_graph_label(graph) for graph in graphs]
    counts = {label: max(1, labels.count(label)) for label in sorted(set(labels))}
    sample_weights = [1.0 / counts[label] for label in labels]
    sampler = WeightedRandomSampler(
        weights=torch.tensor(sample_weights, dtype=torch.float),
        num_samples=len(sample_weights),
        replacement=True,
    )
    return DataLoader(graphs, batch_size=batch_size, sampler=sampler)


def _compute_metrics(labels: List[int], predictions: List[int], probabilities: List[float]) -> Dict[str, Any]:
    _, _, _, accuracy_score, precision_score, recall_score, roc_auc_score = _require_training_deps()
    metrics = {
        "accuracy": float(accuracy_score(labels, predictions)) if labels else 0.0,
        "precision": float(precision_score(labels, predictions, zero_division=0)) if labels else 0.0,
        "recall": float(recall_score(labels, predictions, zero_division=0)) if labels else 0.0,
    }
    if labels and len(set(labels)) > 1:
        metrics["roc_auc"] = float(roc_auc_score(labels, probabilities))
    else:
        metrics["roc_auc"] = None
    return metrics


def evaluate_model(model, graphs: List[Any], batch_size: int = 8) -> Dict[str, Any]:
    torch, _, DataLoader, _, _, _, _ = _require_training_deps()
    loader = DataLoader(graphs, batch_size=batch_size, shuffle=False)
    labels: List[int] = []
    predictions: List[int] = []
    probabilities: List[float] = []
    model.eval()
    device = next(model.parameters()).device
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            logits = model(batch)
            probs = torch.softmax(logits, dim=-1)[:, 1]
            preds = torch.argmax(logits, dim=-1)
            labels.extend(batch.y.view(-1).cpu().tolist())
            predictions.extend(preds.cpu().tolist())
            probabilities.extend(probs.cpu().tolist())
    metrics = _compute_metrics(labels, predictions, probabilities)
    metrics["label_counts"] = {
        "benign": int(sum(1 for label in labels if label == 0)),
        "malicious": int(sum(1 for label in labels if label == 1)),
    }
    return metrics


def train_graph_model(
    *,
    train_graphs: List[Any],
    test_graphs: List[Any],
    output_dir: Path,
    hidden_dim: int = 128,
    dropout: float = 0.1,
    learning_rate: float = 1e-3,
    epochs: int = 10,
    batch_size: int = 8,
    seed: int = 42,
    embedding_model_name: str = "microsoft/codebert-base",
    embedding_pooling: str = "mean",
    classifier_head: str = "mlp",
    graph_feature_flags: Dict[str, Any] | None = None,
    aux_weight: float = 0.0,
    contrastive_weight: float = 0.0,
    encoder_lr: float = 2e-5,
) -> Dict[str, Any]:
    torch, _, _, _, _, _, _ = _require_training_deps()
    if not train_graphs:
        raise ValueError("No training graphs provided.")
    if not test_graphs:
        raise ValueError("No test graphs provided.")

    set_global_seed(seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_dim = _infer_input_dim(train_graphs)
    residual_dim = _infer_residual_dim(train_graphs)
    model_flags = _model_flags(graph_feature_flags)
    model = HeteroGraphClassifier(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        dropout=dropout,
        classifier_head=classifier_head,
        residual_dim=residual_dim,
        aux_node_head=model_flags["aux_node_head"],
        edge_weighted=model_flags["edge_weighted"],
        encoder=model_flags["encoder"],
    ).to(device)

    train_loader = _build_loader(train_graphs, batch_size=batch_size, weighted=True)
    # Encoder (finetuned) params get a smaller LR than the freshly-init GNN.
    encoder_module = getattr(model, "encoder", None)
    encoder_param_ids = {id(p) for p in encoder_module.parameters()} if encoder_module is not None else set()
    base_params = [p for p in model.parameters() if p.requires_grad and id(p) not in encoder_param_ids]
    param_groups = [{"params": base_params, "lr": learning_rate}]
    encoder_train_params = [
        p for p in (encoder_module.parameters() if encoder_module is not None else []) if p.requires_grad
    ]
    if encoder_train_params:
        param_groups.append({"params": encoder_train_params, "lr": encoder_lr})
    optimizer = torch.optim.Adam(param_groups)
    labels = [_graph_label(graph) for graph in train_graphs]
    benign_count = max(1, labels.count(0))
    malicious_count = max(1, labels.count(1))
    class_weights = torch.tensor(
        [len(labels) / (2 * benign_count), len(labels) / (2 * malicious_count)],
        dtype=torch.float32,
        device=device,
    )
    loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights)

    history: List[Dict[str, Any]] = []
    best_score = -1.0
    best_state = None
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        batch_count = 0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            out = model.run(batch)
            loss = loss_fn(out["logits"], batch.y.view(-1))
            if aux_weight > 0.0:
                aux = _aux_node_loss(out, batch, torch)
                if aux is not None:
                    loss = loss + aux_weight * aux
            if contrastive_weight > 0.0:
                con = _supervised_contrastive(out["embedding"], batch.y, torch)
                if con is not None:
                    loss = loss + contrastive_weight * con
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.item())
            batch_count += 1

        train_metrics = evaluate_model(model, train_graphs, batch_size=batch_size)
        test_metrics = evaluate_model(model, test_graphs, batch_size=batch_size)
        history.append(
            {
                "epoch": epoch,
                "train_loss": epoch_loss / max(1, batch_count),
                "train_metrics": train_metrics,
                "test_metrics": test_metrics,
            }
        )
        monitored = test_metrics.get("roc_auc")
        score = float(monitored) if monitored is not None else float(test_metrics.get("accuracy", 0.0))
        if score >= best_score:
            best_score = score
            best_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}

    if best_state is None:
        raise RuntimeError("Training completed without producing a model state.")

    checkpoint_path = output_dir / "model.pt"
    torch.save(best_state, checkpoint_path)
    model.load_state_dict(best_state)
    final_metrics = evaluate_model(model, test_graphs, batch_size=batch_size)
    metadata = {
        "gnn_model_type": "hetero_sage",
        "classifier_head": classifier_head,
        "input_dim": input_dim,
        "residual_dim": residual_dim,
        "node_security_features": bool((graph_feature_flags or {}).get("node_security_features", False)),
        "grounding_residual_feature": bool((graph_feature_flags or {}).get("grounding_residual_feature", False)),
        "edge_weighted_grounding": model_flags["edge_weighted"],
        "aux_node_head": model_flags["aux_node_head"],
        "encoder_finetune": model_flags["encoder"] is not None,
        "encoder_trainable_layers": int((graph_feature_flags or {}).get("encoder_trainable_layers", 2)),
        "encoder_max_len": int((graph_feature_flags or {}).get("encoder_max_len", 128)),
        "encoder_scope": str((graph_feature_flags or {}).get("encoder_scope", "added")),
        "aux_weight": float(aux_weight),
        "contrastive_weight": float(contrastive_weight),
        "hidden_dim": hidden_dim,
        "dropout": dropout,
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "seed": seed,
        "embedding_model_name": embedding_model_name,
        "embedding_pooling": embedding_pooling,
        "train_graph_count": len(train_graphs),
        "test_graph_count": len(test_graphs),
        "metrics": final_metrics,
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    (output_dir / "training_history.json").write_text(json.dumps(history, indent=2, sort_keys=True), encoding="utf-8")
    return metadata
