"""Canonical graph construction for structural misalignment."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np


def build_canonical_graph(
    *,
    instance_id: str,
    graph_label: int,
    subtasks: List[Dict[str, Any]],
    candidate_nodes: List[Dict[str, Any]],
    code_edges: List[Dict[str, Any]],
    links: List[Dict[str, Any]],
    subtask_features: np.ndarray,
    code_features: np.ndarray,
    injection_markers: List[str] | None = None,
) -> Dict[str, Any]:
    subtask_id_to_index = {subtask["subtask_id"]: idx for idx, subtask in enumerate(subtasks)}
    node_id_to_index = {str(node.get("node_id", "")): idx for idx, node in enumerate(candidate_nodes)}

    dependency_edges: List[Dict[str, Any]] = []
    for subtask in subtasks:
        dst_idx = subtask_id_to_index[subtask["subtask_id"]]
        for dependency in subtask.get("depends_on", []):
            if dependency not in subtask_id_to_index:
                continue
            dependency_edges.append(
                {
                    "src": subtask_id_to_index[dependency],
                    "dst": dst_idx,
                    "kind": "depends_on",
                }
            )

    cfg_edges: List[Dict[str, Any]] = []
    for edge in code_edges:
        src = str(edge.get("src", ""))
        dst = str(edge.get("dst", ""))
        if src not in node_id_to_index or dst not in node_id_to_index:
            continue
        cfg_edges.append(
            {
                "src": node_id_to_index[src],
                "dst": node_id_to_index[dst],
                "kind": str(edge.get("kind", "fallthrough")),
            }
        )

    cross_edges: List[Dict[str, Any]] = []
    for link in links:
        subtask_id = str(link.get("subtask_id", ""))
        if subtask_id not in subtask_id_to_index:
            continue
        src_idx = subtask_id_to_index[subtask_id]
        scores = link.get("scores", {}) if isinstance(link.get("scores"), dict) else {}
        for node_id in link.get("node_ids", []):
            if node_id not in node_id_to_index:
                continue
            cross_edges.append(
                {
                    "src": src_idx,
                    "dst": node_id_to_index[node_id],
                    "score": float(scores.get(node_id, 0.0)),
                    "fallback_used": bool(link.get("fallback_used", False)),
                }
            )

    return {
        "instance_id": instance_id,
        "graph_label": int(graph_label),
        "subtasks": subtasks,
        "code_nodes": candidate_nodes,
        "subtask_features": subtask_features.tolist(),
        "code_features": code_features.tolist(),
        "injection_markers": list(injection_markers or []),
        "edges": {
            "subtask_to_subtask": dependency_edges,
            "code_to_code": cfg_edges,
            "subtask_to_code": cross_edges,
        },
    }


def _edge_index(edges: List[Dict[str, Any]]):
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - dependency guard
        raise ImportError("Graph serialization requires torch.") from exc
    if not edges:
        return torch.zeros((2, 0), dtype=torch.long)
    pairs = [[int(edge["src"]), int(edge["dst"])] for edge in edges]
    return torch.tensor(pairs, dtype=torch.long).T.contiguous()


# Fixed-length "ungrounded mass" summary: the core thesis is that injected code
# grounds to no subtask, i.e. its best grounding-edge similarity is low.
GROUNDING_RESIDUAL_DIM = 4
_GROUNDING_LOW_THRESHOLD = 0.35


def _encoder_options(config: Dict[str, Any]):
    """Resolve the encoder-finetune (re-encode in loop) settings, or None."""
    if not bool(config.get("encoder_finetune", False)):
        return None
    return {
        "model_name": str(config.get("embedding_model_name", "microsoft/codebert-base")),
        "max_len": int(config.get("encoder_max_len", 128)),
        "scope": str(config.get("encoder_scope", "added")).strip().lower(),
        "trainable_layers": int(config.get("encoder_trainable_layers", 2)),
    }


def graph_feature_options(config: Dict[str, Any]) -> Dict[str, Any]:
    """Read the (config-gated, default-off) graph augmentation flags.

    These decide what gets attached to each HeteroData at build/load time; the
    matching model-side toggles are derived from the same keys (see
    models.gnn.model_options_from_metadata)."""
    if not isinstance(config, dict):
        config = {}
    return {
        "add_security_features": bool(config.get("node_security_features", False)),
        "add_grounding_residual": bool(config.get("grounding_residual_feature", False)),
        "add_edge_weights": bool(config.get("edge_weighted_grounding", False)),
        "add_node_labels": bool(config.get("node_aux_labels", False)),
        "encoder": _encoder_options(config),
    }


def any_augmentation(opts: Dict[str, Any]) -> bool:
    """True if any augmentation needs a rebuild from graph.json (skip .pt cache)."""
    return bool(
        opts.get("add_security_features")
        or opts.get("add_grounding_residual")
        or opts.get("add_edge_weights")
        or opts.get("add_node_labels")
        or opts.get("encoder")
    )


_MIN_MARKER_LEN = 6


def _injected_node_labels(graph: Dict[str, Any], n_code: int) -> np.ndarray:
    """Per-code-node injection label (1 if the node's snippet contains a synth
    injection marker). Only synth-malicious graphs carry markers; everything
    else returns all-zeros (and is masked out of the aux loss)."""
    labels = np.zeros(n_code, dtype=np.float32)
    markers = [
        str(m).strip()
        for m in (graph.get("injection_markers", []) or [])
        if len(str(m).strip()) >= _MIN_MARKER_LEN
    ]
    if not markers or n_code <= 0:
        return labels
    for idx, node in enumerate(graph.get("code_nodes", []) or []):
        if idx >= n_code:
            break
        snippet = str(node.get("code_snippet", "") or "")
        if snippet and any(marker in snippet for marker in markers):
            labels[idx] = 1.0
    return labels


def _grounding_residual_vector(graph: Dict[str, Any], n_code: int) -> np.ndarray:
    """Per-code-node max grounding similarity -> a small graph-level summary.

    The scores are already produced by the linker and stored on the
    subtask->code edges; the base PyG path throws them away. We recover them
    here as [frac_ungrounded, mean_max, min_max, frac_no_edge]."""
    vec = np.zeros(GROUNDING_RESIDUAL_DIM, dtype=np.float32)
    if n_code <= 0:
        return vec
    max_score = np.zeros(n_code, dtype=np.float32)
    has_edge = np.zeros(n_code, dtype=bool)
    for edge in graph.get("edges", {}).get("subtask_to_code", []) or []:
        dst = int(edge.get("dst", -1))
        if 0 <= dst < n_code:
            has_edge[dst] = True
            max_score[dst] = max(max_score[dst], float(edge.get("score", 0.0)))
    vec[0] = float(np.mean(max_score < _GROUNDING_LOW_THRESHOLD))
    vec[1] = float(np.mean(max_score))
    vec[2] = float(np.min(max_score))
    vec[3] = float(np.mean(~has_edge))
    return vec


from functools import lru_cache


@lru_cache(maxsize=4)
def _get_tokenizer(model_name: str):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(model_name)


def _is_live_node(node: Dict[str, Any], scope: str) -> bool:
    if scope == "all":
        return True
    return str(node.get("change_type", "")).strip().lower() in {"added", "modified"}


def _tokenize_nodes(texts: List[str], model_name: str, max_len: int):
    """Tokenize node texts into (input_ids, attention_mask) tensors aligned to
    the node feature rows, for the re-encode-in-loop (encoder finetune) path."""
    import torch

    if not texts:
        return torch.zeros((0, max_len), dtype=torch.long), torch.zeros((0, max_len), dtype=torch.long)
    tok = _get_tokenizer(model_name)
    enc = tok(texts, padding="max_length", truncation=True, max_length=max_len, return_tensors="pt")
    return enc["input_ids"].to(torch.long), enc["attention_mask"].to(torch.long)


def build_pyg_heterodata(
    graph: Dict[str, Any],
    *,
    add_security_features: bool = False,
    add_grounding_residual: bool = False,
    add_edge_weights: bool = False,
    add_node_labels: bool = False,
    encoder: Dict[str, Any] | None = None,
):
    try:
        import torch
        from torch_geometric.data import HeteroData
    except ImportError as exc:  # pragma: no cover - dependency guard
        raise ImportError("PyG graph construction requires torch-geometric.") from exc

    data = HeteroData()
    subtask_features = np.asarray(graph.get("subtask_features", []), dtype=np.float32)
    code_features = np.asarray(graph.get("code_features", []), dtype=np.float32)
    if subtask_features.size == 0:
        subtask_dim = code_features.shape[1] if code_features.ndim == 2 and code_features.size else 768
        subtask_features = np.zeros((0, subtask_dim), dtype=np.float32)
    if code_features.size == 0:
        code_dim = subtask_features.shape[1] if subtask_features.ndim == 2 and subtask_features.size else 768
        code_features = np.zeros((0, code_dim), dtype=np.float32)

    if add_security_features:
        # Real security features for code nodes; zero-pad subtask (NL) nodes so
        # both node types keep a uniform input dimension for the shared convs.
        from src.baseline.structural_misalignment.graph.security_features import (
            SECURITY_FEATURE_DIM,
            build_security_feature_matrix,
        )

        sec = build_security_feature_matrix(graph.get("code_nodes", []) or [], code_features.shape[0])
        code_features = np.concatenate([code_features, sec], axis=1)
        subtask_pad = np.zeros((subtask_features.shape[0], SECURITY_FEATURE_DIM), dtype=np.float32)
        subtask_features = np.concatenate([subtask_features, subtask_pad], axis=1)

    n_code = code_features.shape[0]
    data["subtask"].x = torch.tensor(subtask_features, dtype=torch.float32)
    data["code"].x = torch.tensor(code_features, dtype=torch.float32)
    ground_edges = graph.get("edges", {}).get("subtask_to_code", []) or []
    data["subtask", "depends_on", "subtask"].edge_index = _edge_index(graph.get("edges", {}).get("subtask_to_subtask", []))
    data["code", "cfg", "code"].edge_index = _edge_index(graph.get("edges", {}).get("code_to_code", []))
    data["subtask", "grounds", "code"].edge_index = _edge_index(ground_edges)
    if add_edge_weights:
        # Recover the grounding similarity scores the base conv discards, so an
        # edge-weight-aware conv can consume them on the grounds relation.
        weights = [float(edge.get("score", 0.0)) for edge in ground_edges]
        data["subtask", "grounds", "code"].edge_attr = torch.tensor(weights, dtype=torch.float32).view(-1, 1)
    data.y = torch.tensor([int(graph.get("graph_label", 0))], dtype=torch.long)
    if add_grounding_residual:
        residual = _grounding_residual_vector(graph, n_code)
        data.grounding_residual = torch.tensor(residual, dtype=torch.float32).view(1, -1)
    if add_node_labels:
        # Per-code-node injection label + a graph-level mask (only synth-malicious
        # graphs carry markers -> only they contribute to the aux loss).
        labels = _injected_node_labels(graph, n_code)
        data["code"].inj_label = torch.tensor(labels, dtype=torch.float32)
        has_labels = 1.0 if (graph.get("injection_markers") and n_code > 0) else 0.0
        data.node_label_mask = torch.tensor([has_labels], dtype=torch.float32)
    if encoder is not None:
        model_name = str(encoder.get("model_name", "microsoft/codebert-base"))
        max_len = int(encoder.get("max_len", 128))
        scope = str(encoder.get("scope", "added"))
        from src.baseline.structural_misalignment.embeddings import serialize_code_node_for_embedding
        from src.baseline.structural_misalignment.grounding.schemas import serialize_subtask_for_embedding

        code_nodes = graph.get("code_nodes", []) or []
        code_texts = [serialize_code_node_for_embedding(node) for node in code_nodes]
        code_live = [_is_live_node(node, scope) for node in code_nodes]
        # Guard against feature/text row mismatch: only encode when aligned.
        if len(code_texts) != n_code:
            code_texts = ["" for _ in range(n_code)]
            code_live = [False for _ in range(n_code)]
        ids, mask = _tokenize_nodes(code_texts, model_name, max_len)
        data["code"].input_ids = ids
        data["code"].attention_mask = mask
        data["code"].live_mask = torch.tensor(code_live, dtype=torch.bool)

        subtasks = graph.get("subtasks", []) or []
        sub_texts = [serialize_subtask_for_embedding(st) for st in subtasks]
        if len(sub_texts) != subtask_features.shape[0]:
            sub_texts = ["" for _ in range(subtask_features.shape[0])]
        s_ids, s_mask = _tokenize_nodes(sub_texts, model_name, max_len)
        data["subtask"].input_ids = s_ids
        data["subtask"].attention_mask = s_mask
        data["subtask"].live_mask = torch.ones(subtask_features.shape[0], dtype=torch.bool)
    data.instance_id = str(graph.get("instance_id", "unknown"))
    return data


def write_graph_artifacts(artifact_dir: Path, graph: Dict[str, Any]) -> Dict[str, str]:
    artifact_dir.mkdir(parents=True, exist_ok=True)
    graph_json = artifact_dir / "graph.json"
    graph_json.write_text(json.dumps(graph, indent=2, sort_keys=True), encoding="utf-8")
    paths = {"graph_json": str(graph_json)}
    try:
        import torch
        hetero = build_pyg_heterodata(graph)
        graph_pt = artifact_dir / "graph.pt"
        torch.save(hetero, graph_pt)
        paths["graph_pt"] = str(graph_pt)
    except ImportError:
        pass
    return paths
