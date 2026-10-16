"""Configuration composition and plant fixtures."""

from __future__ import annotations

from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra

from plant.dynamics import build_plant

CONFIG_DIR = str(Path(__file__).resolve().parents[1] / "configs")


def compose_cfg(overrides=None):
    """Compose the Hydra config with optional override strings."""
    GlobalHydra.instance().clear()
    with initialize_config_dir(version_base=None, config_dir=CONFIG_DIR):
        return compose(config_name="config", overrides=overrides or [])


def build_bundle(overrides=None):
    return build_plant(compose_cfg(overrides))


def nominal_for(bundle, overrides=None):
    """GNC nominal (vehicle/plant constants) for a bundle — used by the allocation tests."""
    from gnc.build import default_nominal

    cfg = compose_cfg(overrides)
    return default_nominal(bundle.params, cfg.gnc.allocation, str(cfg.environment.atmosphere.model))


@pytest.fixture
def bundle():
    return build_bundle()
