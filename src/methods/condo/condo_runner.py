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
    if divergence == "kld":
        from condo import ConDoAdapterKLD

        return ConDoAdapterKLD(transform_type=transform_type, verbose=0)
    if divergence == "mmd":
        from condo import ConDoAdapterMMD

        return ConDoAdapterMMD(
            transform_type=transform_type,
            bootstrap_fraction=float(par.get("bootstrap_fraction", 1.0)),
            n_epochs=int(par.get("n_epochs", 5)),
            learning_rate=float(par.get("learning_rate", 1e-3)),
            verbose=0,
        )
    raise ValueError(f"Unknown divergence: {divergence!r}")


def _read_input(par: dict, meta: dict) -> ad.AnnData:
    sys.path.append(meta["resources_dir"])
    from read_anndata_partial import read_anndata

    rep = par.get("rep", "features")
    if rep == "features":
        return read_anndata(
            par["input"], X="layers/normalized", obs="obs", var="var", uns="uns"
        )
    if rep == "pca":
        return read_anndata(
            par["input"], obs="obs", obsm="obsm", var="var", uns="uns"
        )
    raise ValueError(f"Unknown rep: {rep!r}")


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
    target_batch = counts[0][int(np.argmax(counts[1]))]
    print(f">> Target batch: {target_batch} ({counts[1].max()} cells)", flush=True)

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
