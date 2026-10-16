"""GNC configuration must fail loudly on impossible engine selections."""

import pytest
from conftest import build_bundle, compose_cfg

from gnc.build import build_gnc


def test_invalid_engine_index_is_rejected():
    overrides = ["gnc.scheduler.landing_engines=[999]"]
    cfg = compose_cfg(overrides)
    bundle = build_bundle(overrides)

    with pytest.raises(ValueError, match="landing engine index 999 invalid"):
        build_gnc(cfg, bundle.params)
