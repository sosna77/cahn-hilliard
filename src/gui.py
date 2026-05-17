import numpy as np
import pyqtgraph as pg
from PySide6.QtWidgets import QMainWindow, QHBoxLayout, QVBoxLayout, QWidget, QPushButton, QApplication
from PySide6.QtGui import QTransform
from PySide6.QtCore import Qt
from .config import SimConfig
from .engine import CHWorker

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Cahn-Hilliard Simulation')
        self.resize(1200, 600)

        self.energy_history = []
        self.time_history = []
        self.current_step = 0

        plot_layout = QHBoxLayout()

        self.plot_widget = pg.PlotWidget(title='Magnetisation field m(r)')
        plot_layout.addWidget(self.plot_widget)

        self.plot_item = pg.ImageItem()
        self.plot_widget.addItem(self.plot_item)

        self.lut = pg.HistogramLUTWidget()
        plot_layout.addWidget(self.lut)

        self.lut.setImageItem(self.plot_item)
        self.lut.gradient.loadPreset('viridis')
        self.lut.setLevels(-1.0, 1.0)

        self.energy_widget = pg.PlotWidget(title='Free energy plot F(t)')
        self.energy_widget.setLabel('bottom', 't')
        self.energy_widget.setLabel('left', 'F')
        self.energy_curve = self.energy_widget.plot(pen='r')
        plot_layout.addWidget(self.energy_widget)

        main_layout = QVBoxLayout()
        main_layout.addLayout(plot_layout)

        self.reset_button = QPushButton('RESET')
        self.reset_button.clicked.connect(self.reset_sim)
        main_layout.addWidget(self.reset_button)


        central_widget = QWidget()
        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)

        cfg = SimConfig()
        self.plot_item.setPos(-cfg.L/2, -cfg.L/2)
        transform = QTransform()
        DX = cfg.L / cfg.N
        transform.scale(DX, DX)
        self.plot_item.setTransform(transform)

        # pretty pyqtgraph
        cmap = pg.colormap.getFromMatplotlib('viridis')
        self.plot_item.setColorMap(cmap)
        
        self.plot_widget.setXRange(-cfg.L/2, cfg.L/2)
        self.plot_widget.setYRange(-cfg.L/2, cfg.L/2)
        self.plot_widget.getViewBox().disableAutoRange()
        self.plot_widget.getViewBox().setAspectLocked(True)

        # worker
        self.start_worker()

    def update_screen(self, m):
        self.plot_item.setImage(m.T, autoLevels=False, levels=(-1.0,1.0))

    def update_energy_plot(self, step, energy):
        self.time_history.append(step)
        self.energy_history.append(energy)
        self.energy_curve.setData(self.time_history, self.energy_history)

    def reset_sim(self):
        if self.worker.isRunning():
            self.worker.stop()
            self.worker.wait()

        QApplication.processEvents()

        self.energy_history.clear()
        self.time_history.clear()
        self.current_step = 0
        self.energy_curve.setData([],[])
        self.start_worker()

    def start_worker(self):
        self.cfg = SimConfig()
        self.worker = CHWorker(self.cfg)
        self.worker.frame_ready.connect(self.update_screen)
        self.worker.energy_ready.connect(self.update_energy_plot)
        self.worker.start()

    def closeEvent(self, event):
        if self.worker.isRunning():
            self.worker.stop()
            self.worker.wait()
        event.accept()