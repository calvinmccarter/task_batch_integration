"""Shared runner for the ConDo batch-integration method.

For each non-target batch (paired with the largest batch as target), fits
one ConDo adapter conditioning on cell_type and applies the learned
transform.

Supports three orthogonal axes (set via the ``par`` dict in the viash
script):

* ``divergence``     : 'kld' | 'mmd'
* ``transform_type`` : 'location-scale' | 'affine'
* ``rep``            : 'features' | 'pca'
    - 'features': fit on normalized expression, write corrected_counts.
    - 'pca':      fit on obsm['X_pca'], write obsm['X_emb'] (embedding).
* ``hvg_only`` (features only): restrict fit to var['hvg'] columns; pass
  through non-HVGs at source values.

MMD-specific knobs (``bootstrap_fraction``, ``n_epochs``,
``learning_rate``) are forwarded to ``ConDoAdapterMMD``.
"""
from __future__ import annotations

import sys
from typing import Any

import anndata as ad
import numpy as np
from scipy.sparse import csr_matrix, issparse

# condo 0.8.0 references `np.Inf`, removed in NumPy 2.0. Restore the alias
# before any condo import (used in condo.utils.EarlyStopping). The viash
# config also pins numpy<2 in docker, but the shim makes the runner work
# in newer environments too.
if not hasattr(np, "Inf"):
    np.Inf = np.inf


def _to_dense(x: Any) -> np.ndarray:
    return x.toarray() if issparse(x) else np.asarray(x)


def _build_adapter(par: dict[str, Any]):
    divergence = par["divergence"]
    transform_type = par["transform_type"]
    device = par.get("device", "cpu")
    if divergence == "kld":
        from condo import ConDoAdapterKLD

        return ConDoAdapterKLD(
            transform_type=transform_type, verbose=0, device=device,
        )
    if divergence == "mmd":
        from condo import ConDoAdapterMMD

        kwargs = dict(
            transform_type=transform_type,
            bootstrap_fraction=float(par.get("bootstrap_fraction", 1.0)),
            n_epochs=int(par.get("n_epochs", 5)),
            learning_rate=float(par.get("learning_rate", 1e-3)),
            mmd_size=int(par.get("mmd_size", 20)),
            batch_size=int(par.get("batch_size", 8)),
            weight_decay=float(par.get("weight_decay", 1e-4)),
            random_state=int(par.get("random_state", 42)),
            verbose=0,
            device=device,
            optimizer=str(par.get("optimizer", "adamw")),
        )
        return ConDoAdapterMMD(**kwargs)
    raise ValueError(f"Unknown divergence: {divergence!r}")


def _read_input(par: dict, meta: dict) -> ad.AnnData:
    sys.path.append(meta["resources_dir"])
    from read_anndata_partial import read_anndata

    rep = par.get("rep", "features")
    # The target-selection heuristic needs obsm['X_pca']; pull it in both modes
    # so target_mode='best_pre_asw' works regardless of rep.
    if rep == "features":
        return read_anndata(
            par["input"], X="layers/normalized",
            obs="obs", obsm="obsm", var="var", uns="uns",
        )
    if rep == "pca":
        return read_anndata(
            par["input"], obs="obs", obsm="obsm", var="var", uns="uns"
        )
    raise ValueError(f"Unknown rep: {rep!r}")


def _pick_target_by_pre_asw(
    adata: ad.AnnData, batches: np.ndarray, cell_types: np.ndarray
) -> tuple[str, dict]:
    """Pick the batch with the highest per-batch silhouette of cell_type in
    the precomputed PCA representation. Returns (target_batch_label,
    per_batch_asw_table)."""
    from scib.metrics import silhouette

    if "X_pca" not in adata.obsm:
        raise ValueError(
            "target_mode='best_pre_asw' requires obsm['X_pca']; "
            f"obsm keys present: {list(adata.obsm.keys())}"
        )
    batch_labels = np.unique(batches)
    per_batch: dict[str, float] = {}
    for b in batch_labels:
        mask = batches == b
        if mask.sum() < 4:
            per_batch[b] = float("-inf")
            continue
        sub = adata[mask].copy()
        cts = sub.obs["cell_type"].astype(str)
        keep_cts = cts.value_counts()[cts.value_counts() >= 2].index
        keep_mask = cts.isin(keep_cts).values
        if keep_mask.sum() < 4 or len(keep_cts) < 2:
            per_batch[b] = float("-inf")
            continue
        sub2 = sub[keep_mask].copy()
        try:
            s = float(silhouette(sub2, label_key="cell_type", embed="X_pca"))
        except Exception:
            s = float("-inf")
        per_batch[b] = s
    best = max(per_batch.items(), key=lambda kv: kv[1])
    return best[0], per_batch


def _select_feature_columns(adata: ad.AnnData, hvg_only: bool) -> np.ndarray | None:
    """Return a boolean column mask if HVG-only is requested, else None."""
    if not hvg_only:
        return None
    if "hvg" not in adata.var.columns:
        raise ValueError(
            "--hvg_only requires a boolean 'hvg' column in var; "
            f"available columns: {list(adata.var.columns)}"
        )
    mask = adata.var["hvg"].astype(bool).values
    if mask.sum() == 0:
        raise ValueError("--hvg_only requested but no var['hvg'] entries are True")
    return mask


def run_condo(par: dict, meta: dict) -> None:
    rep = par.get("rep", "features")
    hvg_only = bool(par.get("hvg_only", False)) and rep == "features"

    print(f">> Read input (rep={rep}, hvg_only={hvg_only})", flush=True)
    adata = _read_input(par, meta)

    # condo's product_prior dispatches on dtype.kind ∈ {'U','S'} for discrete
    # confounders; pandas .astype(str).values returns object dtype which
    # slips past that check.
    batches = np.asarray(adata.obs["batch"].astype(str).values, dtype="U")
    cell_types = np.asarray(adata.obs["cell_type"].astype(str).values, dtype="U")

    counts = np.unique(batches, return_counts=True)
    target_override = par.get("target_batch")
    target_mode = par.get("target_mode", "agglomerative")

    # Compute per-batch pre-integration asw — used both by best_pre_asw
    # (as the target-choice criterion) and by agglomerative (as the
    # batch_score for seed selection and neighbor ranking).
    _, per_batch_asw = _pick_target_by_pre_asw(adata, batches, cell_types)

    if target_override is not None:
        if target_override not in set(counts[0]):
            raise ValueError(
                f"target_batch {target_override!r} not in batches: {list(counts[0])}"
            )
        target_batch = target_override
        target_n = int(counts[1][list(counts[0]).index(target_batch)])
        print(
            f">> Target batch (override): {target_batch} ({target_n} cells)",
            flush=True,
        )
    elif target_mode == "best_pre_asw":
        target_batch = max(per_batch_asw, key=lambda b: per_batch_asw[b])
        target_n = int(counts[1][list(counts[0]).index(target_batch)])
        print(
            f">> Target batch (best_pre_asw): {target_batch} "
            f"(pre_asw={per_batch_asw[target_batch]:.4f}, {target_n} cells)",
            flush=True,
        )
        for b, s in sorted(per_batch_asw.items(), key=lambda kv: -kv[1]):
            print(f"    pre_asw[{b}] = {s:.4f}", flush=True)
    elif target_mode == "agglomerative":
        # No single target — agglomerative_integrate walks the
        # compatibility graph from a seed (highest pre_asw) and merges
        # batches one at a time into a growing pool.
        target_batch = None
        print(">> Target mode: agglomerative (seed = argmax pre_asw)", flush=True)
        for b, s in sorted(per_batch_asw.items(), key=lambda kv: -kv[1]):
            print(f"    pre_asw[{b}] = {s:.4f}", flush=True)
    else:
        raise ValueError(f"Unknown target_mode: {target_mode!r}")

    if rep == "features":
        Y_full = _to_dense(adata.X).astype(np.float64)
        hvg_mask = _select_feature_columns(adata, hvg_only)
        if hvg_mask is not None:
            print(
                f">> HVG-only: fitting on {int(hvg_mask.sum())}/{Y_full.shape[1]} genes",
                flush=True,
            )
            Y = Y_full[:, hvg_mask]
        else:
            Y = Y_full
    else:  # pca
        Y_full = None
        hvg_mask = None
        Y = np.asarray(adata.obsm["X_pca"], dtype=np.float64)

    if target_mode == "agglomerative":
        from condo.batch_integration import agglomerative_integrate

        def _adapter_factory():
            return _build_adapter(par)

        result = agglomerative_integrate(
            Y=Y,
            batches=batches,
            confounders=cell_types,
            batch_score=per_batch_asw,
            adapter_factory=_adapter_factory,
            verbose=True,
        )
        Y_out = result.Y_out
        # Skip the fixed-target loop below; jump straight to output.
    else:
        Y_out = Y.copy()

        target_mask = batches == target_batch
        Yt = Y[target_mask]
        Zt = cell_types[target_mask].reshape(-1, 1)

        source_batches = [b for b in counts[0] if b != target_batch]
        for batch in source_batches:
            src_mask = batches == batch
            n_src = int(src_mask.sum())
            Ys = Y[src_mask]
            Zs = cell_types[src_mask].reshape(-1, 1)
            shared = set(np.unique(Zs.ravel())) & set(np.unique(Zt.ravel()))
            print(
                f">> Adapt batch={batch!r} (n={n_src}, shared cell types={len(shared)})",
                flush=True,
            )
            if not shared:
                # ConDo's product_prior requires at least one shared confounder
                # value between source and target — otherwise the conditional
                # weights collapse to 0/0 and the MMD/KLD loss is undefined.
                # When that happens, leave the source batch unchanged (Y_out
                # already holds its original values from the .copy() above).
                print(
                    f">> SKIP batch={batch!r}: no cell types shared with target; "
                    f"cells left unchanged",
                    flush=True,
                )
                continue
            adapter = _build_adapter(par)
            adapter.fit(Ys, Yt, Zs, Zt)
            Y_out[src_mask] = adapter.transform(Ys)

    print(">> Build output", flush=True)
    if rep == "features":
        if hvg_mask is None:
            corrected = Y_out
        else:
            # Start from the original full-feature matrix; overwrite the HVG
            # columns with the adapted values. Non-HVG columns are pass-through.
            corrected = Y_full.copy()
            corrected[:, hvg_mask] = Y_out
        output = ad.AnnData(
            obs=adata.obs[[]],
            var=adata.var[[]],
            layers={"corrected_counts": csr_matrix(corrected)},
            uns={
                "dataset_id": adata.uns["dataset_id"],
                "normalization_id": adata.uns["normalization_id"],
                "method_id": meta["name"],
            },
        )
    else:  # pca -> embedding output
        output = ad.AnnData(
            obs=adata.obs[[]],
            var=adata.var[[]],
            obsm={"X_emb": Y_out},
            uns={
                "dataset_id": adata.uns["dataset_id"],
                "normalization_id": adata.uns["normalization_id"],
                "method_id": meta["name"],
            },
        )

    print(">> Write output", flush=True)
    output.write_h5ad(par["output"], compression="gzip")
