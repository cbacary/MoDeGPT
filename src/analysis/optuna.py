import optuna
from src.adapters.CompressionConfig import CompressionConfig

import os

from src.run_modegpt import main


def objective(trial: optuna.Trial):
    """
    Used for llama3-8B to ensure replicability of original paper.

    Minimize for ppl
    """

    config = CompressionConfig(
        model="meta-llama/Meta-Llama-3-8B",
        device=0,
        calib_size=128,
        calibs_batch_size=16,
        output_dir=f"{os.environ.get('TMPDIR', '/tmp')}/compressed_output/Llama3-8B/",
        note=f"optuna sparsity Llam3-8B trial {trial.number}",
        compression_ratio=0.25,
        order=f"mlp,qk,vo",
        max_sparsity=0.95,
        sparsity_smoothing=trial.suggest_float("sparsity_smoothing", 0.0225, 0.25),
        ridge_vo=trial.suggest_float("ridge_vo", 1e-7, 1e-1, log=True),
        ridge_qk=trial.suggest_float("ridge_qk", 1e-7, 1e-1, log=True),
        dataset="alpaca",
        optuna=True,
    )

    score = main(trial=trial, config=config)

    return score


study = optuna.create_study(
    study_name="modegpt-llama3-8b",
    storage="sqlite:///sweep-modegpt-llama3-8b.db",
    load_if_exists=True,
    direction="minimize",
)

study.optimize(objective, n_trials=20)

print(study.best_params)
print(study.best_value)
