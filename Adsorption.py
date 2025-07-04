# -*- coding: utf-8 -*-
"""
Created on Fri Jul  4 14:21:46 2025

@author: ozgur.ozcan
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import (QApplication, QWidget, QVBoxLayout, QPushButton,
                             QLabel, QComboBox, QLineEdit)
from PyQt5.QtCore import Qt
import imageio

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

class AdsorptionSimulator(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("V2O5 Adsorption & Resistance Simulator")
        self.setGeometry(200, 200, 600, 1000)
        layout = QVBoxLayout()

        # Surface type selection
        self.surface_label = QLabel("Select Surface Type:")
        layout.addWidget(self.surface_label)
        self.surface_combo = QComboBox()
        self.surface_combo.addItems(["nanorod", "nanoflake", "nanoparticle"])
        layout.addWidget(self.surface_combo)

        # SCCM input
        self.sccm_label = QLabel("Gas Flow Rate (SCCM):")
        layout.addWidget(self.sccm_label)
        self.sccm_input = QLineEdit()
        self.sccm_input.setPlaceholderText("e.g. 500")
        layout.addWidget(self.sccm_input)

        # Sticking coefficient input
        self.stick_label = QLabel("Sticking Coefficient (0-1):")
        layout.addWidget(self.stick_label)
        self.stick_input = QLineEdit()
        self.stick_input.setPlaceholderText("e.g. 0.5")
        layout.addWidget(self.stick_input)

        # R0 input
        self.r0_label = QLabel("Baseline Resistance R0 (Ohms):")
        layout.addWidget(self.r0_label)
        self.r0_input = QLineEdit()
        self.r0_input.setPlaceholderText("e.g. 1000")
        layout.addWidget(self.r0_input)

        # k input
        self.k_label = QLabel("Sensitivity Coefficient k:")
        layout.addWidget(self.k_label)
        self.k_input = QLineEdit()
        self.k_input.setPlaceholderText("e.g. 2")
        layout.addWidget(self.k_input)

        # Start button
        self.start_btn = QPushButton("Start Simulation")
        self.start_btn.clicked.connect(self.run_simulation)
        layout.addWidget(self.start_btn)

        # Status
        self.status_label = QLabel("")
        layout.addWidget(self.status_label)

        # Heatmap canvas
        self.figure = Figure(figsize=(5,4))
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)

        # Adsorption graph canvas
        self.figure_ads = Figure(figsize=(5,3))
        self.canvas_ads = FigureCanvas(self.figure_ads)
        layout.addWidget(self.canvas_ads)

        # Resistance graph canvas
        self.figure_res = Figure(figsize=(5,3))
        self.canvas_res = FigureCanvas(self.figure_res)
        layout.addWidget(self.canvas_res)

        self.setLayout(layout)

    def run_simulation(self):
        surface_type = self.surface_combo.currentText()
        try:
            sccm = float(self.sccm_input.text())
            stick_coeff = float(self.stick_input.text())
            r0 = float(self.r0_input.text())
            k = float(self.k_input.text())
            if not (0 <= stick_coeff <=1):
                raise ValueError
        except:
            self.status_label.setText("Please enter valid inputs.")
            return

        self.status_label.setText("Simulation running...")
        QApplication.processEvents()

        # --- Constants ---
        NA = 6.022e23  # Avogadro number
        SCCM_to_mols = 7.45e-7  # mol/s per SCCM at STP

        # --- Simulation parameters ---
        surface_length = 10  # mm
        surface_width = 10   # mm
        surface_thickness = 2  # mm

        total_time = 10  # seconds
        n_frames = 50
        time_per_frame = total_time / n_frames

        mol_per_s = sccm * SCCM_to_mols
        molecule_per_s = mol_per_s * NA
        molecule_per_frame = int(molecule_per_s * time_per_frame * stick_coeff)

        # Adsorption multiplier by morphology
        if surface_type == 'nanorod':
            adsorption_multiplier = 2.0
        elif surface_type == 'nanoflake':
            adsorption_multiplier = 1.2
        elif surface_type == 'nanoparticle':
            adsorption_multiplier = 0.8
        else:
            adsorption_multiplier = 1.0

        filenames = []
        adsorbed_positions = []

        time_points_1s = []
        ads_counts_1s = []
        resistance_1s = []
        current_second = 0

        max_sites = 1e6  # arbitrary total adsorption sites

        for t in range(n_frames):
            effective_molecules = int(molecule_per_frame * adsorption_multiplier)

            x_positions = np.random.uniform(0, surface_length, effective_molecules)
            y_positions = np.random.uniform(0, surface_width, effective_molecules)
            z_positions = np.full(effective_molecules, surface_thickness)

            for i in range(effective_molecules):
                adsorbed_positions.append([x_positions[i], y_positions[i], z_positions[i]])

            elapsed_time = (t+1)*time_per_frame
            if elapsed_time >= current_second + 1:
                current_second += 1
                total_adsorbed = len(adsorbed_positions)
                coverage = min(total_adsorbed / max_sites, 1.0)
                resistance = r0 * np.exp(k * coverage)

                time_points_1s.append(current_second)
                ads_counts_1s.append(total_adsorbed)
                resistance_1s.append(resistance)

            # Heatmap per frame
            ads = np.array(adsorbed_positions)
            heatmap, xedges, yedges = np.histogram2d(
                ads[:,0], ads[:,1], bins=(50,50),
                range=[[0,surface_length],[0,surface_width]]
            )

            fig = plt.figure(figsize=(6,5))
            plt.imshow(
                heatmap.T, origin='lower', extent=[0,surface_length,0,surface_width],
                cmap='hot'
            )
            plt.colorbar(label='Adsorbed Molecule Count')
            plt.title(f"2-CEES Adsorption Heatmap ({surface_type})\nTime: {elapsed_time:.1f} s")
            plt.xlabel("X (mm)")
            plt.ylabel("Y (mm)")

            filename = f"frame_{t}.png"
            plt.savefig(filename)
            filenames.append(filename)
            plt.close()

        # Show final heatmap
        self.figure.clear()
        ax_heat = self.figure.add_subplot(111)
        cax = ax_heat.imshow(
            heatmap.T, origin='lower', extent=[0,surface_length,0,surface_width],
            cmap='hot'
        )
        self.figure.colorbar(cax, ax=ax_heat, label='Adsorbed Molecule Count')
        ax_heat.set_title(f"Final Adsorption Heatmap ({surface_type})")
        ax_heat.set_xlabel("X (mm)")
        ax_heat.set_ylabel("Y (mm)")
        self.canvas.draw()

        # Adsorption vs Time graph
        self.figure_ads.clear()
        ax_ads = self.figure_ads.add_subplot(111)
        ax_ads.plot(time_points_1s, ads_counts_1s, marker='o')
        ax_ads.set_title("Total Adsorbed Molecules vs Time")
        ax_ads.set_xlabel("Time (s)")
        ax_ads.set_ylabel("Adsorbed Molecules")
        self.canvas_ads.draw()

        # Resistance vs Time graph
        self.figure_res.clear()
        ax_res = self.figure_res.add_subplot(111)
        ax_res.plot(time_points_1s, resistance_1s, marker='o', color='green')
        ax_res.set_title("Resistance vs Time")
        ax_res.set_xlabel("Time (s)")
        ax_res.set_ylabel("Resistance (Ohms)")
        self.canvas_res.draw()

        # Save heatmap
        plt.imsave("heatmap.png", heatmap.T, cmap='hot', origin='lower')

        # Create GIF
        with imageio.get_writer('adsorption_simulation.gif', mode='I', duration=0.2) as writer:
            for filename in filenames:
                image = imageio.imread(filename)
                writer.append_data(image)

        self.status_label.setText("✅ Simulation completed. GIF, heatmap, adsorption and resistance graphs generated.")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = AdsorptionSimulator()
    window.show()
    sys.exit(app.exec_())
