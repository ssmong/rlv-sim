"""Monte Carlo worker configuration regressions."""

from pathlib import Path

from conftest import compose_cfg

from scripts.run_montecarlo import _worker_config


def test_salto_aero_table_path_is_absolute_before_spawn():
    cfg = compose_cfg()

    worker_cfg = _worker_config(cfg)

    path = Path(worker_cfg["plant"]["aero"]["file"]["path"])
    assert path.is_absolute()
    assert path.samefile(Path("data/aero/salto_t3_aero.npz"))
    assert cfg.plant.aero.file.path == "data/aero/salto_t3_aero.npz"
