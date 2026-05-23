import sys

## VIASH START
par = {
    "input": "resources_test/task_batch_integration/cxg_immune_cell_atlas/dataset.h5ad",
    "output": "output.h5ad",
    "divergence": "kld",
    "transform_type": "location-scale",
    "rep": "features",
    "hvg_only": False,
    "bootstrap_fraction": 1.0,
    "n_epochs": 5,
    "learning_rate": 1e-3,
}
meta = {
    "name": "condo",
    "resources_dir": ".",
}
## VIASH END

sys.path.append(meta["resources_dir"])
from condo_runner import run_condo

run_condo(par, meta)
