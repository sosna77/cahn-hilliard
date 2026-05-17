import numpy as np
from PySide6.QtCore import QThread, Signal
from numba import njit
from .config import SimConfig
import matplotlib.pyplot as plt

class SimState():
    def __init__(self, cfg: SimConfig):
        # ==== GRID INITIALIZATION ====
        self.cfg = cfg
        amp = min(cfg.noise_amp, min(1.0 - cfg.m0, cfg.m0 + 1.0))
        noise = np.random.uniform(-amp, amp, size=(cfg.N, cfg.N))
        self.m = cfg.m0 + noise

        # ==== REAL AND FOURIER SPACE DISCRETIZATION ====
        self.DX = cfg.L / cfg.N
        x = np.linspace( -cfg.L/2, cfg.L/2, cfg.N, endpoint=False)
        k = np.fft.fftfreq(cfg.N, d=self.DX) * 2 * np.pi
        self.eps = cfg.alpha * self.DX
        
        self.X, self.Y = np.meshgrid(x, x, indexing='ij')
        self.kx, self.ky = np.meshgrid(k, k, indexing='ij')

        # === DE-ALIASING ===
        k_max = np.pi / self.DX
        k_cutoff = (2.0 / 3.0) * k_max
        k_mag = np.sqrt(self.kx**2 + self.ky**2)

        self.dealias_mask = np.where(k_mag < k_cutoff, 1.0, 0.0)

    def calculate_energy(self):
        nonlin = 0.25 * (self.m**2 - 1)**2
        grad_x, grad_y = np.gradient(self.m, self.DX)
        lin = 0.5*self.eps**2 * (grad_x**2 + grad_y**2)

        E_total = np.sum((nonlin + lin) * self.DX**2)

        return E_total

    def calculate_magnetization(self):
        return np.sum(self.m * self.DX**2)

@njit(cache=True)
def calculate_nonlin(m):
    return m**3 - m

@njit(cache=True)
def calculate_new_m(mk, fk, kx, ky, dt, eps):
    k2 = kx**2 + ky**2
    num = mk - dt * k2 * fk
    denom = 1 + dt * (eps**2) * (k2**2)
    return num / denom

# @njit(cache=True)
# def calculate_new_m_stochastic(mk, fk, kx, ky, dt, eps, noise_k):
#     k2 = kx**2 + ky**2
#     k_mag = np.sqrt(k2)

#     stochastic_term = np.sqrt(dt) * k_mag * noise_k

#     num = mk - dt * k2 * fk + stochastic_term
#     denom = 1 + dt * (eps**2) * (k2**2)
#     return num / denom

class Solver():
    def step(self, state: SimState, cfg: SimConfig):
        f = calculate_nonlin(state.m)
        fk = np.fft.fft2(f) * state.dealias_mask            # de-aliasing
        mk = np.fft.fft2(state.m)

        # white noise (thermal)
        noise_real = np.random.normal(0.0, cfg.noise_amp, size=(cfg.N, cfg.N))
        noise_k = np.fft.fft2(noise_real)

        mk_new = calculate_new_m(mk, fk, state.kx, state.ky, cfg.DT, state.eps)
        # mk_new = calculate_new_m_stochastic(mk, fk, state.kx, state.ky, cfg.DT, state.eps, noise_k)
        state.m = np.real(np.fft.ifft2(mk_new))

def main():
    iters = 1000
    cfg = SimConfig()
    solver = Solver()
    state = SimState(cfg)
    energy = np.empty(iters)
    magnetization = np.empty(iters)

    for i in range(iters):
        solver.step(state=state, cfg=cfg)
        energy[i] = state.calculate_energy()
        magnetization[i] = state.calculate_magnetization()
    ts = np.arange(iters)
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    ax[0].plot(ts, energy, label='energy')
    ax[1].plot(ts, magnetization, label='magnet')
    plt.legend()
    plt.show()


class CHWorker(QThread):
    frame_ready = Signal(object)
    energy_ready = Signal(int, float)

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.is_running = True

    def run(self):
        self.cfg = SimConfig()
        self.solver = Solver()
        self.state = SimState(self.cfg)

        # ==== MAIN LOOP ====
        step_count = 0
        while self.is_running:
            self.solver.step(self.state, self.cfg)
            step_count += 1
            if step_count % self.cfg.draw_every == 0:
                self.frame_ready.emit(self.state.m.copy())
            if step_count % 1 == 0:
                energy = self.state.calculate_energy()
                self.energy_ready.emit(step_count, energy)

    def stop(self):
        self.is_running = False
    
if __name__ == '__main__':
    main()



