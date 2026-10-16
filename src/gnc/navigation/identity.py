"""Identity navigation with optional NumPy sensor noise and bias."""

from __future__ import annotations

import numpy as np

from core.frames import quat_mul, quat_normalize
from core.interfaces import Measurement
from core.seeding import fold_in, rng_from_seed


class IdentityNav:
    def __init__(
        self,
        pos_noise: float = 0.0,
        vel_noise: float = 0.0,
        att_noise: float = 0.0,
        rate_noise: float = 0.0,
        mass_noise: float = 0.0,
        pos_bias=0.0,
        vel_bias=0.0,
        att_bias=0.0,
        rate_bias=0.0,
        mass_bias: float = 0.0,
    ):
        """Configure optional independent sensor noise and bias."""
        self.pos_noise = float(pos_noise)
        self.vel_noise = float(vel_noise)
        self.att_noise = float(att_noise)
        self.rate_noise = float(rate_noise)
        self.mass_noise = float(mass_noise)
        self.pos_bias = np.asarray(pos_bias, dtype=float)
        self.vel_bias = np.asarray(vel_bias, dtype=float)
        self.att_bias = np.asarray(att_bias, dtype=float)
        self.rate_bias = np.asarray(rate_bias, dtype=float)
        self.mass_bias = float(mass_bias)
        self._pos_on = bool(pos_noise) or bool(np.any(self.pos_bias != 0))
        self._vel_on = bool(vel_noise) or bool(np.any(self.vel_bias != 0))
        self._att_on = bool(att_noise) or bool(np.any(self.att_bias != 0))
        self._rate_on = bool(rate_noise) or bool(np.any(self.rate_bias != 0))
        self._mass_on = bool(mass_noise) or bool(mass_bias != 0)

    @staticmethod
    def _noise(seed, channel, shape=()):
        return rng_from_seed(fold_in(seed, channel)).normal(size=shape)

    def measure(self, state, seed) -> Measurement:
        r, v = state.r_ned, state.v_ned
        if self._pos_on:
            r = np.asarray(r) + self.pos_noise * self._noise(seed, 0, 3) + self.pos_bias
        if self._vel_on:
            v = np.asarray(v) + self.vel_noise * self._noise(seed, 1, 3) + self.vel_bias

        q, w, m = state.q_bn, state.w_body, state.m
        if self._att_on:
            dtheta = self.att_noise * self._noise(seed, 2, 3) + self.att_bias
            dq = np.concatenate([np.ones(1), 0.5 * np.broadcast_to(dtheta, (3,))])
            q = quat_normalize(quat_mul(q, dq))
        if self._rate_on:
            w = np.asarray(w) + self.rate_noise * self._noise(seed, 3, 3) + self.rate_bias
        if self._mass_on:
            m = float(m) + self.mass_noise * float(self._noise(seed, 4)) + self.mass_bias
        return Measurement(r_ned=r, v_ned=v, q_bn=q, w_body=w, m=m)

    def estimate(self, meas: Measurement, act, memory):
        return meas, memory


def initial_memory():
    return {}


def build(ctx):
    """Factory named by configs/gnc/navigation/identity.yaml."""
    return IdentityNav()
