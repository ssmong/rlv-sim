"""GncContext — the single bundle of resources handed to every GNC module factory.

``gnc.build`` constructs one of these and passes it to the ``build(ctx)`` factory named by
the active module's config (``law._target_``, resolved via ``hydra.utils.instantiate``). A
factory reads only the fields its module needs and ignores the rest — so adding a new
guidance / navigation / control module never requires touching ``gnc/build.py``.

Fields:
    cfg         : this module's resolved config node (cfg.gnc.guidance / .navigation / .control).
                  Read your own scalar parameters from here.
    nominal     : vehicle-constants dict from gnc.build.default_nominal (masses, arms, thrust,
                  allocation options). None when not built yet (navigation).
    params      : PlantParams (numeric vehicle/environment leaves). None for navigation.
    active_engines : (N,) 0/1 array of engines this guidance phase fires. None outside guidance.
    phase       : int Phase tag this guidance instance runs in. -1 outside guidance.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class GncContext:
    cfg: Any
    nominal: dict | None = None
    params: Any = None
    active_engines: Any = None
    phase: int = -1
