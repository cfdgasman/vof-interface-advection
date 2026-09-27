"""VOF benchmarks: Rider-Kothe single vortex and Zalesak's slotted disk. Writes tables, figures and GIFs."""

import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter

from vof import advect, circle, face_velocities, rotation_psi, shape_error, single_vortex_psi, slotted_disk

T_VORTEX = 8.0


def vortex_velocity(n):
    h = 1.0 / n
    return lambda t: face_velocities(single_vortex_psi(n, t, T_VORTEX), h)


def rotation_velocity(n):
    h = 1.0 / n
    uv = face_velocities(rotation_psi(n), h)
    return lambda t: uv


def contour(ax, F, **kw):
    n = F.shape[0]
    c = (np.arange(n) + 0.5) / n
    return ax.contour(c, c, F.T, levels=[0.5], **kw)


def study_vortex():
    print("Single vortex, T = 8 (stretch, then exact reversal)")
    print("| N | PLIC error | order | upwind error | mass error (PLIC) | min F | max F − 1 | time |")
    print("|---|---|---|---|---|---|---|---|")
    prev = None
    for n in (32, 64, 128, 256):
        h = 1.0 / n
        F0 = circle(n)
        t0 = time.perf_counter()
        F = advect(F0, vortex_velocity(n), T_VORTEX, h)
        dt = time.perf_counter() - t0
        Fu = advect(F0, vortex_velocity(n), T_VORTEX, h, scheme="upwind")
        e = shape_error(F, F0, h)
        order = f"{np.log2(prev / e):.2f}" if prev else "–"
        prev = e
        mass = abs(F.sum() - F0.sum()) / F0.sum()
        print(f"| {n} | {e:.2e} | {order} | {shape_error(Fu, F0, h):.2e} | {mass:.0e} | "
              f"{F.min():.0e} | {F.max() - 1:.0e} | {dt:.1f} s |")


def figure_vortex(n=128):
    h = 1.0 / n
    F0 = circle(n)
    snaps = {}

    def grab(t, F):
        for target in (2.0, 4.0, 6.0):
            if target not in snaps and t >= target - 1e-9:
                snaps[target] = F.copy()

    F = advect(F0, vortex_velocity(n), T_VORTEX, h, callback=grab)
    Fu = advect(F0, vortex_velocity(n), T_VORTEX, h, scheme="upwind")
    fig, axes = plt.subplots(1, 5, figsize=(16, 3.6))
    panels = [("t = 0", F0), ("t = T/4", snaps[2.0]), ("t = T/2 (max. stretch)", snaps[4.0]),
              ("t = T, PLIC", F), (f"t = T, upwind (max F = {Fu.max():.2f})", Fu)]
    for ax, (title, G) in zip(axes, panels):
        ax.imshow(G.T, origin="lower", extent=(0, 1, 0, 1), cmap="Blues", vmin=0, vmax=1)
        contour(ax, G, colors="k", linewidths=1)
        ax.set(title=title, xticks=[], yticks=[])
    for ax in axes[3:]:
        contour(ax, F0, colors="r", linewidths=1, linestyles="--")
    axes[3].plot([], [], "k-", label="F = 0.5")
    axes[3].plot([], [], "r--", label="exact")
    axes[3].legend(fontsize=7, loc="lower left")
    fig.suptitle(f"Rider–Kothe single vortex, {n}×{n}, F = 0.5 contour")
    fig.tight_layout()
    fig.savefig("docs/single_vortex.png", dpi=110)
    plt.close(fig)


def figure_zalesak(n=200):
    h = 1.0 / n
    F0 = slotted_disk(n)
    F = advect(F0, rotation_velocity(n), 1.0, h)
    Fu = advect(F0, rotation_velocity(n), 1.0, h, scheme="upwind")
    print(f"\nZalesak, {n}x{n}, one revolution: PLIC error {shape_error(F, F0, h):.2e}, "
          f"upwind error {shape_error(Fu, F0, h):.2e}, PLIC mass error {abs(F.sum() - F0.sum()) / F0.sum():.0e}")
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    for ax, G, name in ((axes[0], F, "Geometric PLIC-VOF"), (axes[1], Fu, "Algebraic upwind")):
        ax.imshow(G.T, origin="lower", extent=(0, 1, 0, 1), cmap="Blues", vmin=0, vmax=1)
        contour(ax, F0, colors="r", linewidths=1, linestyles="--")
        contour(ax, G, colors="k", linewidths=1.2)
        ax.set(xlim=(0.3, 0.7), ylim=(0.55, 0.95), xticks=[], yticks=[],
               title=f"{name}\nerror = {shape_error(G, F0, h):.2e}")
    fig.suptitle(f"Zalesak's slotted disk after one revolution ({n}×{n}); red dashed = exact")
    fig.tight_layout()
    fig.savefig("docs/zalesak.png", dpi=110)
    plt.close(fig)


def animate(name, F0, velocity, t_end, n, frames=60, zoom=None):
    h = 1.0 / n
    shots = [(0.0, F0.copy())]
    step = t_end / frames

    def grab(t, F):
        if t >= len(shots) * step - 1e-9:
            shots.append((t, F.copy()))

    advect(F0, velocity, t_end, h, callback=grab)
    fig, ax = plt.subplots(figsize=(3.6, 3.6))
    fig.subplots_adjust(0, 0, 1, 0.92)
    im = ax.imshow(F0.T, origin="lower", extent=(0, 1, 0, 1), cmap="Blues", vmin=0, vmax=1)
    ax.set(xticks=[], yticks=[])
    if zoom:
        ax.set(xlim=zoom[0], ylim=zoom[1])
    title = ax.set_title("")
    lines = []

    def update(k):
        t, F = shots[k]
        im.set_data(F.T)
        for c in lines:
            c.remove()
        lines.clear()
        lines.append(contour(ax, F, colors="k", linewidths=0.8))
        title.set_text(f"{name}   t = {t:.2f}")
        return [im]

    anim = FuncAnimation(fig, update, frames=len(shots))
    anim.save(f"docs/{name.split()[0].lower()}.gif", writer=PillowWriter(fps=15), dpi=70)
    plt.close(fig)


def main():
    import sys

    if "--figures-only" in sys.argv:
        figure_vortex()
        return
    study_vortex()
    figure_vortex()
    figure_zalesak()
    n = 128
    animate("Vortex (PLIC)", circle(n), vortex_velocity(n), T_VORTEX, n, frames=80)
    animate("Zalesak (PLIC)", slotted_disk(n), rotation_velocity(n), 1.0, n, frames=60)


if __name__ == "__main__":
    main()
