"""Geometric Volume-of-Fluid (PLIC) interface advection on a uniform 2D grid.

* Interface reconstruction: Youngs' normal (gradient of F from a 3x3 stencil)
  and the analytic line-constant / volume relations of Scardovelli & Zaleski.
* Advection: directionally split, geometric (Lagrangian-free) fluxes of the
  PLIC polygon, with the Weymouth & Yue (2010) dilation correction. This keeps
  volume exactly conserved and 0 <= F <= 1 when CFL <= 1/2.
* The split order alternates every step (x-y, then y-x) for second order in time.
* For comparison: an algebraic first-order upwind scheme with the same splitting.

Layout: F[i, j] is cell (x_i, y_j) on the unit square. Face velocities come
from a stream function at cell corners, so the discrete divergence is zero.
"""

from __future__ import annotations

import numpy as np

EPS = 1e-12


# ---------------------------------------------------------------- geometry


def volume_fraction(nx, ny, c):
    """Fraction of the unit square where nx*x + ny*y <= c (vectorised)."""
    nx, ny, c = np.broadcast_arrays(np.asarray(nx, float), np.asarray(ny, float), np.asarray(c, float))
    c = c - np.minimum(nx, 0.0) - np.minimum(ny, 0.0)  # reflect onto nx, ny >= 0
    ax, ay = np.abs(nx), np.abs(ny)
    s = ax + ay
    safe = s > EPS
    s_ = np.where(safe, s, 1.0)
    m1 = np.minimum(ax, ay) / s_
    m2 = 1.0 - m1
    a = np.clip(c / s_, 0.0, 1.0)
    flip = a > 0.5
    a = np.where(flip, 1.0 - a, a)
    with np.errstate(divide="ignore", invalid="ignore"):
        v = np.where(a < m1, a * a / (2.0 * m1 * m2), (a - 0.5 * m1) / m2)
    v = np.where(flip, 1.0 - v, v)
    return np.where(safe, v, np.where(c >= 0, 1.0, 0.0))


def line_constant(nx, ny, f):
    """Inverse of volume_fraction: c such that the unit cell holds volume f."""
    ax, ay = np.abs(nx), np.abs(ny)
    s = np.maximum(ax + ay, EPS)
    m1 = np.minimum(ax, ay) / s
    m2 = 1.0 - m1
    flip = f > 0.5
    v = np.where(flip, 1.0 - f, f)
    a = np.where(v < 0.5 * m1 / np.maximum(m2, EPS), np.sqrt(2.0 * m1 * m2 * v), v * m2 + 0.5 * m1)
    a = np.where(flip, 1.0 - a, a)
    return a * s + np.minimum(nx, 0.0) + np.minimum(ny, 0.0)


def youngs_normal(F):
    """n = -grad(F) from corner gradients averaged to cell centres (Youngs 1982)."""
    P = np.pad(F, 1, mode="edge")
    gx = P[1:, 1:] + P[1:, :-1] - P[:-1, 1:] - P[:-1, :-1]  # at corners, (N+1, N+1)
    gy = P[1:, 1:] + P[:-1, 1:] - P[1:, :-1] - P[:-1, :-1]
    nx = -0.25 * (gx[1:, 1:] + gx[1:, :-1] + gx[:-1, 1:] + gx[:-1, :-1])
    ny = -0.25 * (gy[1:, 1:] + gy[1:, :-1] + gy[:-1, 1:] + gy[:-1, :-1])
    norm = np.hypot(nx, ny)
    ok = norm > EPS
    norm = np.where(ok, norm, 1.0)
    return nx / norm, ny / norm


# ---------------------------------------------------------------- velocity


def face_velocities(psi, h):
    """u on x-faces (N+1, N) and v on y-faces (N, N+1) from corner stream function psi (N+1, N+1)."""
    u = (psi[:, 1:] - psi[:, :-1]) / h
    v = -(psi[1:, :] - psi[:-1, :]) / h
    return u, v


def corners(n):
    x = np.linspace(0.0, 1.0, n + 1)
    return np.meshgrid(x, x, indexing="ij")


def single_vortex_psi(n, t, period):
    X, Y = corners(n)
    return np.sin(np.pi * X) ** 2 * np.sin(np.pi * Y) ** 2 * np.cos(np.pi * t / period) / np.pi


def rotation_psi(n, omega=2 * np.pi):
    X, Y = corners(n)
    return -0.5 * omega * ((X - 0.5) ** 2 + (Y - 0.5) ** 2)


# ---------------------------------------------------------------- advection


def _sweep_plic(F, c_flux, axis, dil):
    """One directional PLIC sweep. c_flux: face Courant numbers along `axis` (N+1 faces)."""
    nx, ny = youngs_normal(F)
    if axis == 1:  # treat y like x by swapping components and transposing
        F, nx, ny, c_flux, dil = F.T, ny.T, nx.T, c_flux.T, dil.T
    c_line = line_constant(nx, ny, F)
    mixed = (F > EPS) & (F < 1 - EPS)
    oriented = np.hypot(nx, ny) > 0.5  # a mixed cell with no usable normal is treated as uniform
    cf = c_flux[1:-1]  # interior faces between cell i and i+1
    # donor for positive velocity: cell i, strip x in [1 - c, 1]
    cp = np.maximum(cf, 0.0)
    fl_p = volume_fraction(nx[:-1] * cp, ny[:-1], c_line[:-1] - nx[:-1] * (1 - cp)) * cp
    fl_p = np.where(oriented[:-1], fl_p, F[:-1] * cp)
    fl_p = np.where(mixed[:-1], fl_p, np.where(F[:-1] >= 1 - EPS, cp, 0.0))
    # donor for negative velocity: cell i+1, strip x in [0, |c|]
    cm = np.maximum(-cf, 0.0)
    fl_m = volume_fraction(nx[1:] * cm, ny[1:], c_line[1:]) * cm
    fl_m = np.where(oriented[1:], fl_m, F[1:] * cm)
    fl_m = np.where(mixed[1:], fl_m, np.where(F[1:] >= 1 - EPS, cm, 0.0))
    flux = np.zeros_like(c_flux)
    flux[1:-1] = np.where(cf > 0, fl_p, -fl_m)
    Fn = F - (flux[1:] - flux[:-1]) + dil * (c_flux[1:] - c_flux[:-1])
    Fn = np.clip(Fn, 0.0, 1.0)
    return Fn.T if axis == 1 else Fn


def _sweep_upwind(F, c_flux, axis, dil):
    if axis == 1:
        F, c_flux, dil = F.T, c_flux.T, dil.T
    cf = c_flux[1:-1]
    flux = np.zeros_like(c_flux)
    flux[1:-1] = np.where(cf > 0, cf * F[:-1], cf * F[1:])
    Fn = F - (flux[1:] - flux[:-1]) + dil * (c_flux[1:] - c_flux[:-1])
    return Fn.T if axis == 1 else Fn


SCHEMES = {"plic": _sweep_plic, "upwind": _sweep_upwind}


def advect(F, velocity, t_end, h, cfl=0.5, scheme="plic", callback=None):
    """Advance F to t_end. `velocity(t)` returns face velocities (u, v)."""
    sweep = SCHEMES[scheme]
    F = F.copy()
    t = 0.0
    step = 0
    while t < t_end - 1e-12:
        u, v = velocity(t)
        umax = max(np.abs(u).max(), np.abs(v).max(), 1e-12)
        dt = min(cfl * h / umax, t_end - t)
        while True:
            # the CFL bound must hold for the midpoint velocity actually used,
            # which matters when the flow speed changes in time (e.g. the vortex)
            u, v = velocity(t + 0.5 * dt)
            umid = max(np.abs(u).max(), np.abs(v).max())
            if umid * dt <= cfl * h * (1 + 1e-12):
                break
            dt = cfl * h / umid
        cx, cy = u * dt / h, v * dt / h
        dil = (F > 0.5).astype(float)  # Weymouth & Yue: fixed for the whole step
        order = (0, 1) if step % 2 == 0 else (1, 0)
        for axis in order:
            F = sweep(F, cx if axis == 0 else cy, axis, dil)
        t += dt
        step += 1
        if callback is not None:
            callback(t, F)
    return F


# ---------------------------------------------------------------- initial shapes


def _sample(n, inside, sub=8):
    """Cell volume fractions of a shape by sub-cell sampling."""
    s = (np.arange(n * sub) + 0.5) / (n * sub)
    X, Y = np.meshgrid(s, s, indexing="ij")
    return inside(X, Y).reshape(n, sub, n, sub).mean(axis=(1, 3))


def circle(n, xc=0.5, yc=0.75, r=0.15):
    return _sample(n, lambda X, Y: (X - xc) ** 2 + (Y - yc) ** 2 <= r * r)


def slotted_disk(n, xc=0.5, yc=0.75, r=0.15, slot_w=0.05, slot_h=0.25):
    def inside(X, Y):
        disk = (X - xc) ** 2 + (Y - yc) ** 2 <= r * r
        slot = (np.abs(X - xc) <= slot_w / 2) & (Y <= yc - r + slot_h)
        return disk & ~slot

    return _sample(n, inside)


def shape_error(F, F0, h):
    """L1 shape error  E = sum |F - F0| h^2."""
    return np.abs(F - F0).sum() * h * h
