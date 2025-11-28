"""Lightweight wrapper around the hierarchical MARL (HSD) implementation.

This module wires up three conveniences:
- A small agent wrapper that exposes `assign_roles` and `act` for inference.
- A thin training helper that runs `train_hsd.train_function` from the alg/ folder.
- A saving routine that copies the trained checkpoint (decoder, low- and high-level
  Q-networks) into a stable location under `hierarchical_marl/saved`.

The actual algorithm code lives in `hierarchical_marl/alg`.  We simply make it easy
to call into it from the rest of the project without touching sys.path elsewhere.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence


ALG_DIR = Path(__file__).parent / "alg"
RESULTS_DIR = ALG_DIR.parent / "results"
# Keep the existing folder name to avoid surprising the rest of the project.
DEFAULT_SAVE_DIR = Path(__file__).parent / "saved"
DEFAULT_CONFIG = ALG_DIR / "config.json"


def _ensure_alg_path() -> None:
    """Make sure alg/ is importable."""
    alg_path = str(ALG_DIR.resolve())
    if alg_path not in sys.path:
        sys.path.append(alg_path)


def _load_config(config_path: Optional[Path] = None) -> dict:
    cfg_path = Path(config_path) if config_path else DEFAULT_CONFIG
    with open(cfg_path, "r") as f:
        return json.load(f)


@dataclass(frozen=True)
class EnvSpec:
    """Minimal environment specification needed to construct HSD networks."""

    n_agents: int
    state_dim: int
    obs_dim: int
    action_dim: int


class HMARLAgent:
    """Inference-only wrapper around the HSD algorithm."""

    def __init__(self, alg, epsilon: float = 0.0):
        self.alg = alg
        self.epsilon = epsilon

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint: Path,
        env_spec: EnvSpec,
        config_path: Optional[Path] = None,
        device=None,
    ) -> "HMARLAgent":
        """Load a saved HSD checkpoint and return a ready-to-use agent."""
        _ensure_alg_path()
        import alg_hsd  # type: ignore

        config = _load_config(config_path)
        alg_cfg = config["alg"]
        h_cfg = config["h_params"]
        nn_cfg = config["nn_hsd"]

        alg = alg_hsd.Alg(
            alg_cfg,
            h_cfg,
            env_spec.n_agents,
            env_spec.state_dim,
            env_spec.obs_dim,
            env_spec.action_dim,
            h_cfg["N_roles"],
            nn_cfg,
            device=device,
        )
        alg.load(str(checkpoint))
        return cls(alg)

    def assign_roles(self, observations: Sequence, n_roles_current: Optional[int] = None, epsilon: Optional[float] = None):
        """Choose a role for each agent using the high-level policy."""
        n_roles = n_roles_current or getattr(self.alg, "l_z", None)
        if n_roles is None:
            raise ValueError("Unable to infer number of roles; pass n_roles_current explicitly.")
        eps = self.epsilon if epsilon is None else epsilon
        return self.alg.assign_roles(observations, eps, n_roles)

    def act(self, observations: Sequence, roles_one_hot, epsilon: Optional[float] = None):
        """Select primitive actions for all agents given current roles."""
        eps = self.epsilon if epsilon is None else epsilon
        return self.alg.run_actor(observations, roles_one_hot, eps)


def train_and_save(config_path: Optional[Path] = None, save_dir: Optional[Path] = None) -> Path:
    """Run HSD training and copy the resulting checkpoint into `save_dir`.

    Args:
        config_path: Optional path to a JSON config. Falls back to alg/config.json.
        save_dir: Where to copy the final checkpoint and config. Defaults to `saved/`.

    Returns:
        Path to the copied checkpoint.
    """
    save_root = Path(save_dir) if save_dir else DEFAULT_SAVE_DIR
    save_root.mkdir(parents=True, exist_ok=True)

    _ensure_alg_path()
    import train_hsd  # type: ignore

    config = _load_config(config_path)
    cwd = Path.cwd()
    try:
        # train_hsd uses relative paths (../results), so run from alg/
        os.chdir(ALG_DIR)
        train_hsd.train_function(config)
    finally:
        os.chdir(cwd)

    dir_name = config["main"]["dir_name"]
    model_name = config["main"]["model_name"]

    src_dir = RESULTS_DIR / dir_name
    src_model = src_dir / model_name
    if not src_model.exists():
        raise FileNotFoundError(f"Expected model at {src_model}, but it was not created.")

    dest_model = save_root / model_name
    shutil.copy2(src_model, dest_model)

    src_cfg = src_dir / "config.json"
    if src_cfg.exists():
        shutil.copy2(src_cfg, dest_model.with_suffix(".config.json"))

    return dest_model


# --- Lightweight shims to plug into the shared runner without pulling in RMAPP O machinery. ---
class HMARLPolicyShim:
    """Minimal policy stub so the shared runner can construct a policy object."""

    def __init__(self, *args, **kwargs):
        pass

    def lr_decay(self, *_args, **_kwargs):
        pass

    def get_actions(self, *args, **kwargs):
        raise RuntimeError("HMARLPolicyShim does not support online collection. Use hmarl_wrapper.train_and_save.")

    def act(self, *args, **kwargs):
        raise RuntimeError("HMARLPolicyShim does not support online collection. Use hmarl_wrapper.train_and_save.")


class HMARLTrainerShim:
    """Minimal trainer stub to satisfy the runner constructor."""

    def __init__(self, _all_args, policy, device=None):
        self.policy = policy
        self.device = device

    def prep_rollout(self):
        pass

    def train(self, *_args, **_kwargs):
        raise RuntimeError("HMARLTrainerShim does not support online updates. Use hmarl_wrapper.train_and_save.")


def _parse_args():
    import argparse

    parser = argparse.ArgumentParser(description="Train HSD (HMARL) and save checkpoint.")
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="Path to alg/config.json compatible with train_hsd.py.",
    )
    parser.add_argument(
        "--save-dir",
        type=Path,
        default=DEFAULT_SAVE_DIR,
        help="Directory to copy the final checkpoint into.",
    )
    return parser.parse_args()


def main():
    args = _parse_args()
    ckpt = train_and_save(config_path=args.config, save_dir=args.save_dir)
    print(f"Saved HMARL checkpoint to {ckpt}")


if __name__ == "__main__":
    main()
