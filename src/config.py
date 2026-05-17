from dataclasses import dataclass

@dataclass
class SimConfig:
    N: int = 512                # grid is N x N
    L: float = 10.0             # length of the window, DX = L / N

    DT: float = 5e-4            # time step

    m0: float = 0.0             # avg magnetization
    noise_amp: float = 0.4      # initial thermal noise around avg magnetization

    alpha: float = 1.5          # parameter in Cahn-Hilliard equation eps = alpha * DX
    draw_every: int = 1         # how often to update the diagram 