import numpy as np
import pytest

from vof import advect, circle, face_velocities, rotation_psi, shape_error, single_vortex_psi, slotted_disk
from vof.plic import line_constant, volume_fraction


def test_line_constant_inverts_volume_fraction():
    rng = np.random.default_rng(0)
    theta = rng.uniform(0, 2 * np.pi, 5000)
    f = rng.uniform(0, 1, 5000)
    nx, ny = np.cos(theta), np.sin(theta)
    assert np.abs(volume_fraction(nx, ny, line_constant(nx, ny, f)) - f).max() < 1e-12


def test_volume_fraction_simple_cases():
    assert volume_fraction(1.0, 0.0, 0.3) == pytest.approx(0.3)  # vertical line x = 0.3
    assert volume_fraction(1.0, 1.0, 1.0) == pytest.approx(0.5)  # diagonal
    assert volume_fraction(-1.0, 0.0, -0.3) == pytest.approx(0.7)  # x >= 0.3


def _vortex(n, scheme="plic"):
    h = 1.0 / n
    F0 = circle(n)
    vel = lambda t: face_velocities(single_vortex_psi(n, t, 8.0), h)  # noqa: E731
    return F0, advect(F0, vel, 8.0, h, scheme=scheme), h


def test_mass_conservation_and_boundedness():
    F0, F, _ = _vortex(32)
    assert abs(F.sum() - F0.sum()) / F0.sum() < 1e-13
    assert F.min() >= 0.0 and F.max() <= 1.0


def test_plic_converges_and_beats_upwind():
    F0, F, h = _vortex(32)
    e32 = shape_error(F, F0, h)
    F0, F, h = _vortex(64)
    e64 = shape_error(F, F0, h)
    assert np.log2(e32 / e64) > 1.8
    Fu = _vortex(64, "upwind")[1]
    assert e64 < 0.2 * shape_error(Fu, F0, h)


def test_rigid_rotation_preserves_slotted_disk():
    n = 64
    h = 1.0 / n
    F0 = slotted_disk(n)
    uv = face_velocities(rotation_psi(n), h)
    F = advect(F0, lambda t: uv, 1.0, h)
    assert abs(F.sum() - F0.sum()) < 1e-12
    assert shape_error(F, F0, h) < 5e-3
