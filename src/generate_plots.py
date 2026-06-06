import sys
import os
from dataclasses import replace     # do klonowania dataclassy z nadpisaniem pól

import numpy as np
import matplotlib
matplotlib.use("Agg")               # backend bez okna (headless)
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# ── Importy z projektu ────────────────────────────────────────────────
try:
    from src.config import SimConfig
    from src.engine import SimState, Solver
    print("Zaimportowano z pakietu src/")
except ModuleNotFoundError:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import importlib, types
    pkg = types.ModuleType("src")
    pkg.__path__ = [os.path.dirname(os.path.abspath(__file__))]
    sys.modules["src"] = pkg

    spec_cfg = importlib.util.spec_from_file_location(
        "src.config",
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.py")
    )
    mod_cfg = importlib.util.module_from_spec(spec_cfg)
    sys.modules["src.config"] = mod_cfg
    spec_cfg.loader.exec_module(mod_cfg)

    spec_eng = importlib.util.spec_from_file_location(
        "src.engine",
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "engine.py")
    )
    mod_eng = importlib.util.module_from_spec(spec_eng)
    sys.modules["src.engine"] = mod_eng
    spec_eng.loader.exec_module(mod_eng)

    SimConfig = mod_cfg.SimConfig
    SimState  = mod_eng.SimState
    Solver    = mod_eng.Solver
    print("Zaimportowano z katalogu bieżącego (fallback)")

# ── Katalog wyjściowy ─────────────────────────────────────────────────
OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "output")
os.makedirs(OUT_DIR, exist_ok=True)

CMAP = plt.cm.RdBu_r   # czerwony = +1 (spin up), niebieski = -1 (spin down)

# ── Funkcja symulacji ─────────────────────────────────────────────────
def run_scenario(m0: float,
                 steps: int   = 4000,
                 save_every: int = 40,
                 N: int  = 256,
                 **cfg_overrides) -> tuple:
    cfg = SimConfig(m0=m0, N=N, **cfg_overrides)
    state  = SimState(cfg)
    solver = Solver()

    frames, energy_hist, time_hist = [], [], []

    print(f"  Symulacja m0={m0:+.2f}  N={N}  steps={steps} ...", flush=True)
    for i in range(steps):
        solver.step(state, cfg)
        if i % save_every == 0:
            frames.append(state.m.copy())
            energy_hist.append(float(state.calculate_energy()))
            time_hist.append(i * cfg.DT)

    return frames, energy_hist, time_hist, cfg


# ── Funkcje rysujące ──────────────────────────────────────────────────
def save_snapshots(frames, time_hist, cfg, label: str):
    """5 klatek równomiernie rozłożonych w czasie → jeden plik PNG."""
    idx = [0,
           len(frames) // 4,
           len(frames) // 2,
           3 * len(frames) // 4,
           len(frames) - 1]

    fig, axes = plt.subplots(1, len(idx), figsize=(14, 3))
    fig.patch.set_facecolor("#1a1a2e")
    for ax in axes:
        ax.set_facecolor("#1a1a2e")

    for col, s in enumerate(idx):
        im = axes[col].imshow(
            frames[s], cmap=CMAP, vmin=-1, vmax=1,
            origin="lower", interpolation="bilinear"
        )
        axes[col].set_title(f"t = {time_hist[s]:.2f}", color="white", fontsize=9)
        axes[col].axis("off")

    fig.suptitle(
        f"Ewolucja pola magnetyzacji  $m_0 = {cfg.m0}$  "
        f"($N={cfg.N}$, $\\alpha={cfg.alpha}$, $\\Delta t={cfg.DT}$)",
        color="white", fontsize=11, y=1.03
    )
    cb = fig.colorbar(im, ax=axes, fraction=0.015, pad=0.02)
    cb.ax.yaxis.set_tick_params(color="white")
    plt.setp(cb.ax.yaxis.get_ticklabels(), color="white")

    path = os.path.join(OUT_DIR, f"snapshots_{label}.png")
    plt.savefig(path, dpi=120, bbox_inches="tight", facecolor="#1a1a2e")
    plt.close()
    print(f"    Zapisano: {path}")


def save_energy_plot(energy_hist, time_hist, cfg, label: str):
    """Wykres F(t) → PNG."""
    fig, ax = plt.subplots(figsize=(6, 3))
    fig.patch.set_facecolor("#1a1a2e")
    ax.set_facecolor("#1a1a2e")

    ax.plot(time_hist, energy_hist, color="#e94560", lw=1.5)
    ax.set_xlabel("$t$", color="white")
    ax.set_ylabel("$\\mathcal{F}$", color="white")
    ax.set_title(
        f"Energia swobodna  $m_0 = {cfg.m0}$",
        color="white"
    )
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_color("#555555")

    plt.tight_layout()
    path = os.path.join(OUT_DIR, f"energy_{label}.png")
    plt.savefig(path, dpi=120, bbox_inches="tight", facecolor="#1a1a2e")
    plt.close()
    print(f"    Zapisano: {path}")


def save_gif(frames, cfg, label: str, fps: int = 12):
    """Animacja wszystkich zebranych klatek → GIF."""
    fig, ax = plt.subplots(figsize=(5, 5))
    fig.patch.set_facecolor("black")
    ax.axis("off")

    im = ax.imshow(
        frames[0], cmap=CMAP, vmin=-1, vmax=1,
        origin="lower", animated=True, interpolation="bilinear"
    )
    ax.set_title(f"$m_0 = {cfg.m0}$", color="white", fontsize=11)

    def _update(i):
        im.set_data(frames[i])
        return [im]

    ani = animation.FuncAnimation(
        fig, _update, frames=len(frames), interval=1000 // fps, blit=True
    )
    path = os.path.join(OUT_DIR, f"anim_{label}.gif")
    ani.save(path, writer="pillow", fps=fps, dpi=80)
    plt.close()
    print(f"    Zapisano: {path}")


def generate_all(m0: float, label: str, **run_kwargs):
    """Odpalenie symulacji + zapis wszystkich trzech typów plików."""
    print(f"\n[{label}]  m0 = {m0:+.2f}")
    frames, energy_hist, time_hist, cfg = run_scenario(m0=m0, **run_kwargs)
    save_snapshots(frames, time_hist, cfg, label)
    save_energy_plot(energy_hist, time_hist, cfg, label)
    save_gif(frames, cfg, label)


# ── Uruchomienie ──────────────────────────────────────────────────────
if __name__ == "__main__":
    np.random.seed(42)


    generate_all( 0.00, "m0_0",   steps=4000, save_every=40, N=256)
    generate_all(+0.35, "m0_pos", steps=4000, save_every=40, N=256)
    generate_all(-0.35, "m0_neg", steps=4000, save_every=40, N=256)

    print(f"\nGotowe! Pliki zapisane w: {OUT_DIR}/")
