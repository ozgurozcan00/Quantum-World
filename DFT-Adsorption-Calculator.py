#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author : Özgür Özcan
"""

import sys, os, re, math, csv, tempfile, subprocess, shutil, bisect, random
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict

from PyQt5 import QtCore, QtGui, QtWidgets

# ------------------------------ Birim dönüşüm ------------------------------ #
RY2EV = 13.605693009
BOHR2A = 0.529177210903
NA = 6.02214076e23   # 1/mol
R_gas = 8.314462618  # J/mol/K

# -------------------- Basit element renk/yarıçap tabloları ---------------- #
COV_RAD = {
    'H':0.31,'C':0.76,'N':0.71,'O':0.66,'F':0.57,'P':1.07,'S':1.05,
    'Cl':1.02,'Cu':1.32,'Ni':1.24,'Pt':1.36,'Au':1.36,'Ag':1.45,
    'Si':1.11,'Al':1.21,'Fe':1.24,'Co':1.18,'Ti':1.36,'V':1.34,
    'Cr':1.22,'Mn':1.22,'B':0.85,'Na':1.66,'K':2.03,'Mg':1.41,'Zn':1.22
}
COLORS = {
    'H': QtGui.QColor(255,255,255), 'C': QtGui.QColor(80,80,80), 'N': QtGui.QColor(48,80,248),
    'O': QtGui.QColor(255,13,13), 'F': QtGui.QColor(144,224,80), 'P': QtGui.QColor(255,128,0),
    'S': QtGui.QColor(255,255,48), 'Cl': QtGui.QColor(31,240,31), 'Si': QtGui.QColor(240,200,160),
    'Cu': QtGui.QColor(200,120,60), 'Ni': QtGui.QColor(80,208,80), 'Pt': QtGui.QColor(208,208,224),
    'Au': QtGui.QColor(255,209,35), 'Ag': QtGui.QColor(192,192,192), 'Al': QtGui.QColor(191,166,166),
    'Fe': QtGui.QColor(224,102,51), 'Co': QtGui.QColor(0,102,153), 'Ti': QtGui.QColor(153,153,255),
    'V': QtGui.QColor(166,166,222), 'Cr': QtGui.QColor(138,153,199), 'Mn': QtGui.QColor(156,122,199),
    'Na': QtGui.QColor(171,92,242), 'K': QtGui.QColor(143,64,212), 'Mg': QtGui.QColor(138,255,0),
    'Zn': QtGui.QColor(125,128,176), 'B': QtGui.QColor(255,181,181)
}
def elem_color(sym: str) -> QtGui.QColor:
    return COLORS.get(sym, QtGui.QColor(200,200,200))
def cov_rad(sym: str) -> float:
    return COV_RAD.get(sym, 1.0)

# ------------------------------- Veri yapıları ----------------------------- #
@dataclass
class Frame:
    symbols: List[str]
    positions: List[Tuple[float,float,float]]
    cell: Optional[List[List[float]]] = None

@dataclass
class StepMetrics:
    step: int
    total_energy_eV: Optional[float] = None
    max_force: Optional[float] = None
    scf_acc_Ry: Optional[float] = None
    fermi_eV: Optional[float] = None
    magnetization: Optional[float] = None

# --------------------------------- Parser ---------------------------------- #
class QEParser:
    RE_CELL = re.compile(r'^\s*CELL_PARAMETERS\s*\((?P<u>[A-Za-z]+)\)\s*$', re.I)
    RE_ALAT = re.compile(r'^\s*alat\s*=\s*([\-0-9\.Ee\+]+)\s*a0', re.I)
    RE_POS = re.compile(r'^\s*ATOMIC_POSITIONS\s*\((?P<u>[A-Za-z]+)\)\s*$', re.I)
    RE_ENE = re.compile(r'!\s*total energy\s*=\s*([\-0-9\.Ee\+]+)\s*Ry', re.I)
    RE_FERMI = re.compile(r'the Fermi energy is\s*([\-0-9\.Ee\+]+)\s*eV', re.I)
    RE_SCFACC = re.compile(r'estimated scf accuracy\s*<\s*([\-0-9\.Ee\+]+)\s*Ry', re.I)
    RE_MAG = re.compile(r'total magnetization\s*=\s*([\-0-9\.Ee\+]+)', re.I)
    RE_FORCE = re.compile(r'(?:Total force|max force)\s*=\s*([\-0-9\.Ee\+]+)', re.I)
    RE_ATLINE = re.compile(r'^\s*([A-Za-z][a-z]?)\s+([\-0-9\.Ee\+]+)\s+([\-0-9\.Ee\+]+)\s+([\-0-9\.Ee\+]+)')

    def parse_file(self, path: str) -> Tuple[List[Frame], List[StepMetrics], Optional[float]]:
        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()

        frames: List[Frame] = []
        metrics: List[StepMetrics] = []
        step = -1
        last_energy_ev = None
        current_cell = None
        alat_bohr = None

        i = 0
        n = len(lines)
        while i < n:
            ln = lines[i].rstrip('\n')

            m = self.RE_ALAT.search(ln)
            if m:
                try: alat_bohr = float(m.group(1))
                except: alat_bohr = None
                i += 1; continue

            m = self.RE_ENE.search(ln)
            if m:
                val_ry = float(m.group(1)); last_energy_ev = val_ry * RY2EV
                step += 1
                metrics.append(StepMetrics(step=step, total_energy_eV=last_energy_ev))
                i += 1; continue

            m = self.RE_SCFACC.search(ln)
            if m and step >= 0:
                metrics[-1].scf_acc_Ry = float(m.group(1))
                i += 1; continue

            m = self.RE_FERMI.search(ln)
            if m and step >= 0:
                metrics[-1].fermi_eV = float(m.group(1))
                i += 1; continue

            m = self.RE_MAG.search(ln)
            if m and step >= 0:
                metrics[-1].magnetization = float(m.group(1))
                i += 1; continue

            m = self.RE_FORCE.search(ln)
            if m and step >= 0:
                metrics[-1].max_force = float(m.group(1))
                i += 1; continue

            m = self.RE_CELL.match(ln)
            if m:
                unit = m.group('u').lower()
                mat = []
                for k in range(1,4):
                    if i+k >= n: break
                    parts = lines[i+k].split()
                    if len(parts) >= 3:
                        try: mat.append([float(parts[0]), float(parts[1]), float(parts[2])])
                        except: pass
                if len(mat) == 3:
                    if unit.startswith('angstrom'):
                        current_cell = mat
                    elif unit.startswith('alat'):
                        scale = (alat_bohr or 1.0) * BOHR2A
                        current_cell = [[c*scale for c in row] for row in mat]
                    elif unit.startswith('bohr') or unit.startswith('au'):
                        current_cell = [[c*BOHR2A for c in row] for row in mat]
                i += 4; continue

            m = self.RE_POS.match(ln)
            if m:
                unit = m.group('u').lower()
                j = i+1
                syms, coords = [], []
                while j < n:
                    mm = self.RE_ATLINE.match(lines[j])
                    if not mm: break
                    syms.append(mm.group(1))
                    x,y,z = float(mm.group(2)), float(mm.group(3)), float(mm.group(4))
                    coords.append((x,y,z)); j += 1
                if syms and coords:
                    if unit.startswith('crystal'):
                        if current_cell is not None:
                            coords = [self.frac_to_cart(p, current_cell) for p in coords]
                    elif unit.startswith('bohr'):
                        coords = [(p[0]*BOHR2A, p[1]*BOHR2A, p[2]*BOHR2A) for p in coords]
                    frames.append(Frame(symbols=syms, positions=coords, cell=current_cell))
                i = j; continue

            i += 1

        return frames, metrics, last_energy_ev

    @staticmethod
    def frac_to_cart(fr: Tuple[float,float,float], cell: List[List[float]]) -> Tuple[float,float,float]:
        ax, ay, az = cell[0], cell[1], cell[2]
        x = fr[0]*ax[0] + fr[1]*ay[0] + fr[2]*az[0]
        y = fr[0]*ax[1] + fr[1]*ay[1] + fr[2]*az[1]
        z = fr[0]*ax[2] + fr[1]*ay[2] + fr[2]*az[2]
        return (x,y,z)

# ------------------------------ Yardımcılar -------------------------------- #
def project_iso(p, elev_deg=20.0, azim_deg=35.0, scale=1.0, center=(0,0,0)):
    x,y,z = p[0]-center[0], p[1]-center[1], p[2]-center[2]
    az = math.radians(azim_deg); el = math.radians(elev_deg)
    cx, sx = math.cos(az), math.sin(az)
    x2 = cx*x - sx*y
    y2 = sx*x + cx*y
    z2 = z
    ce, se = math.cos(el), math.sin(el)
    x3 = x2
    y3 = ce*y2 - se*z2
    return (x3*scale, y3*scale)

def bbox3(pts: List[Tuple[float,float,float]]):
    xs = [p[0] for p in pts]; ys = [p[1] for p in pts]; zs = [p[2] for p in pts]
    return (min(xs),max(xs)),(min(ys),max(ys)),(min(zs),max(zs))

def compute_bonds(symbols, coords, scale=1.15, min_cut=0.2):
    bonds = []
    n = len(coords)
    for i in range(n):
        ri = cov_rad(symbols[i])
        for j in range(i+1, n):
            rj = cov_rad(symbols[j])
            cutoff = scale*(ri + rj)
            dx = coords[i][0]-coords[j][0]
            dy = coords[i][1]-coords[j][1]
            dz = coords[i][2]-coords[j][2]
            d = math.sqrt(dx*dx+dy*dy+dz*dz)
            if min_cut < d < cutoff:
                bonds.append((i,j))
    return bonds

def cross(a,b):
    return (a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0])

def norm(v): return math.sqrt(v[0]*v[0]+v[1]*v[1]+v[2]*v[2])

def cell_base_area(cell: Optional[List[List[float]]]) -> Optional[float]:
    if not cell: return None
    a1, a2 = cell[0], cell[1]
    c = cross(a1,a2); return norm(c)

def top_layer_indices(frame: Frame, frac_top: float = 0.15) -> List[int]:
    zvals = [p[2] for p in frame.positions]
    if not zvals: return []
    zmax, zmin = max(zvals), min(zvals)
    span = (zmax - zmin) if zmax>zmin else 1.0
    thresh = zmax - frac_top*span
    return [i for i,p in enumerate(frame.positions) if p[2] >= thresh]

def adsorption_geometry_stats(frame: Frame, anchor_idx: Optional[int] = None, neighbor_cut_extra: float = 0.2) -> Dict[str, object]:
    sym, pos = frame.symbols, frame.positions
    if not pos: return {}
    surf = top_layer_indices(frame)
    if not surf: return {}
    if anchor_idx is None:
        anchor_idx = max(range(len(pos)), key=lambda i: pos[i][2])
    anchor = pos[anchor_idx]
    zsurf_mean = sum(pos[i][2] for i in surf)/len(surf)
    h = anchor[2] - zsurf_mean
    neigh = []
    for j in surf:
        cut = 1.15*(cov_rad(sym[anchor_idx]) + cov_rad(sym[j])) + neighbor_cut_extra
        d = math.dist(anchor, pos[j])
        if d < cut: neigh.append((j, d))
    neigh.sort(key=lambda x: x[1])
    cn = len(neigh)
    avg_d = sum(d for _,d in neigh)/cn if cn>0 else None
    return {"anchor_idx": anchor_idx, "h": h, "coordination": cn, "nearest": neigh[:6], "avg_bond": avg_d}

# ============================ Tema yardımcıları ============================ #
def color_from_choice(name: str) -> QtGui.QColor:
    name = (name or "").lower()
    if "beyaz" in name or "white" in name: return QtGui.QColor(255,255,255)
    if "gri" in name or "gray" in name or "grey" in name: return QtGui.QColor(200,200,200)
    return QtGui.QColor(20,22,28)  # siyah (koyu)

def text_from_choice(name: str) -> QtGui.QColor:
    name = (name or "").lower()
    if "siyah" in name or "black" in name: return QtGui.QColor(10,10,10)
    if "gri" in name or "gray" in name or "grey" in name: return QtGui.QColor(60,60,60)
    return QtGui.QColor(235,235,235)  # beyaz (açık)

# ------------------------------ Görüntüleyici ------------------------------ #
class MoleculeView(QtWidgets.QWidget):
    """Pan/zoom/rotate özellikli izometrik görüntüleyici; tema destekli."""
    requestStatus = QtCore.pyqtSignal(str)
    def __init__(self, parent=None):
        super().__init__(parent)
        self.frames: List[Frame] = []
        self.idx = 0
        self.timer = QtCore.QTimer(self); self.timer.timeout.connect(self.next_frame)
        self.interval = 400
        # kamera
        self.elev = 20.0; self.azim = 35.0
        # çizim opsiyonları
        self.draw_cell = True; self.draw_bonds = True
        self.bond_scale = 1.15; self.bond_width = 2.0
        self.atom_radius_scale = 1.0; self.downsample = 1
        # etkileşim
        self._zoom = 1.0; self._pan = QtCore.QPointF(0,0)
        self._dragging_pan = False; self._dragging_rot = False
        self._last = QtCore.QPointF(0,0)
        # tema
        self.bg_color = QtGui.QColor(20,22,28)
        self.fg_color = QtGui.QColor(230,230,230)
        self.grid_color = QtGui.QColor(120,120,140)
        self.setMinimumSize(420,320)
        self.setMouseTracking(True)
        self.setFocusPolicy(QtCore.Qt.StrongFocus)
        self.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        self.customContextMenuRequested.connect(self._context_menu)

    def set_theme(self, bg: QtGui.QColor, fg: QtGui.QColor):
        self.bg_color = QtGui.QColor(bg)
        self.fg_color = QtGui.QColor(fg)
        g = 0.5*(bg.red()+bg.green()+bg.blue())/255.0
        gc = int(80 + (160 if g<0.5 else -40))
        self.grid_color = QtGui.QColor(gc,gc,gc)
        self.update()

    def set_frames(self, frames: List[Frame]):
        self.frames = frames or []; self.idx = 0
        self.fit_to_view(); self.update()

    def set_interval(self, ms: int):
        self.interval = max(20, ms)
        if self.timer.isActive(): self.timer.start(self.interval)

    def play(self):
        if self.frames: self.timer.start(self.interval); self.requestStatus.emit("Oynatılıyor")

    def pause(self): self.timer.stop(); self.requestStatus.emit("Duraklatıldı")

    def next_frame(self): 
        if not self.frames: return
        self.idx = (self.idx + 1) % len(self.frames); self.update()
    def prev_frame(self):
        if not self.frames: return
        self.idx = (self.idx - 1) % len(self.frames); self.update()

    def keyPressEvent(self, ev: QtGui.QKeyEvent):
        if ev.key() == QtCore.Qt.Key_R:
            self.elev, self.azim = 20.0, 35.0; self._zoom = 1.0; self._pan = QtCore.QPointF(0,0); self.update()
        elif ev.key() == QtCore.Qt.Key_Space:
            if self.timer.isActive(): self.pause()
            else: self.play()
        else:
            super().keyPressEvent(ev)

    def wheelEvent(self, ev: QtGui.QWheelEvent):
        delta_steps = ev.angleDelta().y()/120.0
        if ev.modifiers() & QtCore.Qt.ShiftModifier:
            factor = 1.05 ** delta_steps
            self.bond_scale = max(0.5, min(3.0, self.bond_scale * factor))
            self.requestStatus.emit(f"Bond cutoff scale = {self.bond_scale:.3f}")
            self.update()
        else:
            factor = 1.15 ** delta_steps
            self._zoom = max(0.05, min(100.0, self._zoom*factor)); self.update()

    def mousePressEvent(self, ev: QtGui.QMouseEvent):
        if ev.button() == QtCore.Qt.LeftButton and not (ev.modifiers() & QtCore.Qt.ControlModifier):
            self._dragging_rot = True; self._last = ev.localPos(); ev.accept(); return
        if ev.button() == QtCore.Qt.RightButton or (ev.button() == QtCore.Qt.LeftButton and ev.modifiers() & QtCore.Qt.ControlModifier):
            self._dragging_pan = True; self._last = ev.localPos(); ev.accept(); return
        super().mousePressEvent(ev)

    def mouseMoveEvent(self, ev: QtGui.QMouseEvent):
        if self._dragging_rot:
            delta = ev.localPos() - self._last; self._last = ev.localPos()
            self.azim = (self.azim + delta.x()*0.4) % 360.0
            self.elev = max(-89.0, min(89.0, self.elev - delta.y()*0.3))
            self.update(); return
        if self._dragging_pan:
            delta = ev.localPos() - self._last; self._pan += delta; self._last = ev.localPos(); self.update(); return
        super().mouseMoveEvent(ev)

    def mouseReleaseEvent(self, ev: QtGui.QMouseEvent):
        if ev.button() == QtCore.Qt.LeftButton: self._dragging_rot = False
        if ev.button() == QtCore.Qt.RightButton or (ev.button() == QtCore.Qt.LeftButton and ev.modifiers() & QtCore.Qt.ControlModifier):
            self._dragging_pan = False
        super().mouseReleaseEvent(ev)

    def mouseDoubleClickEvent(self, ev: QtGui.QMouseEvent):
        self.fit_to_view(); self.update(); super().mouseDoubleClickEvent(ev)

    def _context_menu(self, pos):
        m = QtWidgets.QMenu(self)
        act_fit = m.addAction("Sığdır"); act_reset = m.addAction("Reset (kamera)")
        act_origin = m.addAction("Orijin merkezle (pan=0)"); m.addSeparator()
        act_azp = m.addAction("Azimuth +15°"); act_azm = m.addAction("Azimuth −15°")
        act_elp = m.addAction("Elevation +10°"); act_elm = m.addAction("Elevation −10°")
        act = m.exec_(self.mapToGlobal(pos))
        if act == act_fit: self.fit_to_view()
        elif act == act_reset: self.elev, self.azim = 20.0, 35.0; self._zoom=1.0; self._pan=QtCore.QPointF(0,0)
        elif act == act_origin: self._pan=QtCore.QPointF(0,0)
        elif act == act_azp: self.azim = (self.azim + 15.0) % 360.0
        elif act == act_azm: self.azim = (self.azim - 15.0) % 360.0
        elif act == act_elp: self.elev = max(-89.0, min(89.0, self.elev + 10.0))
        elif act == act_elm: self.elev = max(-89.0, min(89.0, self.elev - 10.0))
        self.update()

    def paintEvent(self, e):
        p = QtGui.QPainter(self); p.fillRect(self.rect(), self.bg_color)
        if not self.frames:
            p.setPen(self.fg_color); p.drawText(self.rect(), QtCore.Qt.AlignCenter, "Kare yok"); return

        fr = self.frames[self.idx]
        symbols, pos = fr.symbols, fr.positions
        idxs = list(range(0, len(symbols), max(1, self.downsample)))
        symbols = [symbols[i] for i in idxs]; pos = [pos[i] for i in idxs]

        (xmin,xmax),(ymin,ymax),(zmin,zmax) = bbox3(pos)
        cx, cy, cz = (0.5*(xmin+xmax), 0.5*(ymin+ymax), 0.5*(zmin+zmax))
        span = max(xmax-xmin, ymax-ymin, zmax-zmin, 1e-6)

        margin = 40; w = self.width()-2*margin; h = self.height()-2*margin
        base_scale = 0.8*min(w,h)/span if span>0 else 1.0
        scale = base_scale * self._zoom
        tx = margin + self._pan.x(); ty = margin + h/2 + self._pan.y()

        # Hücre kutusu
        if self.draw_cell and fr.cell:
            a1,a2,a3 = fr.cell[0], fr.cell[1], fr.cell[2]
            origin = (0,0,0)
            corners = [
                origin, a1, a2, a3,
                (a1[0]+a2[0], a1[1]+a2[1], a1[2]+a2[2]),
                (a1[0]+a3[0], a1[1]+a3[1], a1[2]+a3[2]),
                (a2[0]+a3[0], a2[1]+a3[1], a2[2]+a3[2]),
                (a1[0]+a2[0]+a3[0], a1[1]+a2[1]+a3[1], a1[2]+a2[2]+a3[2]),
            ]
            edges = [(0,1),(0,2),(0,3),(1,4),(1,5),(2,4),(2,6),(3,5),(3,6),(4,7),(5,7),(6,7)]
            p.setPen(QtGui.QPen(self.grid_color, 1))
            for i,j in edges:
                x1,y1 = project_iso(corners[i], self.elev, self.azim, scale, (cx,cy,cz))
                x2,y2 = project_iso(corners[j], self.elev, self.azim, scale, (cx,cy,cz))
                p.drawLine(int(tx+x1), int(ty-y1), int(tx+x2), int(ty-y2))

        # Bağlar
        if self.draw_bonds:
            bonds = compute_bonds(symbols, pos, scale=self.bond_scale)
            pen_bond = QtGui.QPen(QtGui.QColor(170,170,170) if self.fg_color.lightness()>120 else QtGui.QColor(210,210,210), max(1.0, float(self.bond_width)))
            p.setPen(pen_bond)
            for i,j in bonds:
                x1,y1 = project_iso(pos[i], self.elev, self.azim, scale, (cx,cy,cz))
                x2,y2 = project_iso(pos[j], self.elev, self.azim, scale, (cx,cy,cz))
                p.drawLine(int(tx+x1), int(ty-y1), int(tx+x2), int(ty-y2))

        # Atomlar
        order = sorted(range(len(pos)), key=lambda k: pos[k][2])
        for k in order:
            sym,coord = symbols[k], pos[k]
            x,y = project_iso(coord, self.elev, self.azim, scale, (cx,cy,cz))
            R = int(4 + 6*self.atom_radius_scale*cov_rad(sym))
            c = elem_color(sym)
            brush = QtGui.QBrush(c)
            pen = QtGui.QPen(QtGui.QColor(20,20,20) if self.bg_color.lightness()<128 else QtGui.QColor(30,30,30)); pen.setWidth(1)
            p.setPen(pen); p.setBrush(brush)
            cxp = int(tx + x); cyp = int(ty - y)
            p.drawEllipse(cxp-R, cyp-R, 2*R, 2*R)

        p.setPen(self.fg_color)
        p.drawText(10, 18, f"Frame {self.idx+1}/{len(self.frames)} | zoom={self._zoom:.2f} | az={self.azim:.1f}° el={self.elev:.1f}° | bondScale={self.bond_scale:.2f}")

    def fit_to_view(self):
        self._zoom = 1.0; self._pan = QtCore.QPointF(0,0)

    def save_current_png(self, path: str):
        img = QtGui.QImage(self.size(), QtGui.QImage.Format_ARGB32); img.fill(self.bg_color)
        qp = QtGui.QPainter(img); self.render(qp); qp.end(); img.save(path)

    def render_to_qimage(self, scale_factor: float = 1.0) -> QtGui.QImage:
        w = max(100, int(self.width()*scale_factor))
        h = max(100, int(self.height()*scale_factor))
        img = QtGui.QImage(w, h, QtGui.QImage.Format_ARGB32); img.fill(self.bg_color)
        qp = QtGui.QPainter(img); qp.scale(scale_factor, scale_factor); self.render(qp); qp.end(); return img

# -------------------------- Çoklu Seri Çizici (UI) ------------------------- #
def palette_colors():
    base = [
        QtGui.QColor(0,200,255), QtGui.QColor(255,140,0), QtGui.QColor(120,220,120),
        QtGui.QColor(180,180,255), QtGui.QColor(255,200,30), QtGui.QColor(255,100,180),
        QtGui.QColor(160,240,240), QtGui.QColor(240,160,160), QtGui.QColor(140,220,180),
        QtGui.QColor(200,200,200)
    ]
    out = []
    for i in range(50):
        c = base[i % len(base)]
        r,g,b,a = c.red(), c.green(), c.blue(), c.alpha()
        dr = random.randint(-10,10); dg = random.randint(-10,10); db = random.randint(-10,10)
        out.append(QtGui.QColor(max(0,min(255,r+dr)),max(0,min(255,g+dg)),max(0,min(255,b+db)),a))
    return out

class MultiLinePlot(QtWidgets.QWidget):
    """Basit çok serili çizici (legend, tooltip, PNG export, tema destekli)."""
    def __init__(self, parent=None, title=""):
        super().__init__(parent)
        self.title = title
        self.series = []  # list of dict: {"xs":[], "ys":[], "color":QColor, "label":str}
        self._margin = 46
        self.setMouseTracking(True)
        self._last_pos = None
        self.show_legend = True
        self._xrange = None; self._yrange = None
        self._hint_lines = []
        self.setMinimumHeight(160)
        self.bg_color = QtGui.QColor(30,32,38)
        self.fg_color = QtGui.QColor(220,220,220)
        self.ax_color = QtGui.QColor(120,120,120)

    def set_theme(self, bg: QtGui.QColor, fg: QtGui.QColor):
        self.bg_color = QtGui.QColor(bg); self.fg_color = QtGui.QColor(fg)
        g = 0.5*(bg.red()+bg.green()+bg.blue())/255.0
        ax = int(90 + (100 if g<0.5 else -40))
        self.ax_color = QtGui.QColor(ax,ax,ax); self.update()

    def set_series_list(self, series_list: List[Dict]):
        self.series = series_list or []; self.update()

    def set_single_series(self, xs: List[float], ys: List[Optional[float]], color: QtGui.QColor, label: str):
        xs2, ys2 = [], []
        for x,y in zip(xs, ys):
            if y is None or (isinstance(y,float) and (y!=y)): continue
            xs2.append(float(x)); ys2.append(float(y))
        self.series = [ {"xs":xs2, "ys":ys2, "color":color, "label":label} ]
        self.update()

    def set_markers(self, markers: List[Tuple[float,str,QtGui.QColor]]):
        self._hint_lines = markers or []; self.update()

    def set_ranges(self, xr=None, yr=None):
        self._xrange = xr; self._yrange = yr; self.update()

    def mouseMoveEvent(self, e: QtGui.QMouseEvent):
        self._last_pos = e.pos(); self.update()
        if not self.series: return
        rect = self.rect(); margin = self._margin
        area = QtCore.QRect(margin, margin, rect.width()-2*margin, rect.height()-2*margin)
        if area.width()<=0 or area.height()<=0: return
        xmin,xmax,ymin,ymax = self._data_bounds()
        if xmax<=xmin or ymax<=ymin: return
        x = xmin + (e.x()-area.left())/max(1,area.width())*(xmax-xmin)
        vals = []
        for s in self.series:
            xs, ys = s["xs"], s["ys"]
            if not xs: continue
            j = bisect.bisect_left(xs, x)
            if j<=0: yv = ys[0]
            elif j>=len(xs): yv = ys[-1]
            else:
                yv = ys[j] if abs(xs[j]-x) < abs(x-xs[j-1]) else ys[j-1]
            vals.append((s["label"], yv))
        tip = "x = {:.3f}\n".format(x) + "\n".join("{}: {:.5f}".format(lbl, v) for lbl,v in vals if isinstance(v,(int,float)))
        QtWidgets.QToolTip.showText(e.globalPos(), tip, self)

    def leaveEvent(self, e):
        self._last_pos = None; self.update()

    def paintEvent(self, e):
        p = QtGui.QPainter(self); rect = self.rect()
        p.fillRect(rect, self.bg_color); p.setRenderHint(QtGui.QPainter.Antialiasing, True)
        margin = self._margin
        area = QtCore.QRect(margin, margin, rect.width()-2*margin, rect.height()-2*margin)
        if area.width()<=0 or area.height()<=0:
            p.setPen(self.fg_color); p.drawText(rect, QtCore.Qt.AlignCenter, "Alan küçük"); return

        p.setPen(self.ax_color); p.drawRect(area)

        if not self.series or not any(s["xs"] for s in self.series):
            p.setPen(self.fg_color); p.drawText(rect, QtCore.Qt.AlignCenter, "Veri yok")
            self._draw_title(p); return

        xmin,xmax,ymin,ymax = self._data_bounds()
        if xmax<=xmin: xmax=xmin+1.0
        if ymax<=ymin: ymax=ymin+1.0
        def mapx(x): return area.left() + (x-xmin)/(xmax-xmin) * area.width()
        def mapy(y): return area.bottom() - (y-ymin)/(ymax-ymin) * area.height()

        for x,label,color in self._hint_lines:
            p.setPen(QtGui.QPen(color, 1, QtCore.Qt.DashLine))
            X = int(mapx(x)); p.drawLine(X, area.top(), X, area.bottom())
            p.setPen(QtGui.QPen(color)); p.drawText(X+4, area.top()+14, label)

        for s in self.series:
            xs, ys, col = s["xs"], s["ys"], s["color"]
            if len(xs) < 2: continue
            p.setPen(QtGui.QPen(col, 2))
            for i in range(1, len(xs)):
                p.drawLine(int(mapx(xs[i-1])), int(mapy(ys[i-1])), int(mapx(xs[i])), int(mapy(ys[i])))

        self._draw_title(p)
        p.setPen(QtGui.QPen(self.fg_color))
        p.drawText(area.left(), area.bottom()+28, f"{xmin:.2f}  …  {xmax:.2f}")
        p.drawText(area.left()-44, area.top()+12, f"{ymax:.3f}")
        p.drawText(area.left()-44, area.bottom(), f"{ymin:.3f}")

        if self.show_legend:
            x0 = area.right()-180; y0 = area.top()+8
            for k,s in enumerate(self.series[:12]):
                p.setPen(QtGui.QPen(s["color"], 3))
                p.drawLine(x0, y0+14*k, x0+18, y0+14*k)
                p.setPen(QtGui.QPen(self.fg_color))
                p.drawText(x0+24, y0+14*k+4, s["label"])

        if self._last_pos and area.contains(self._last_pos):
            p.setPen(QtGui.QPen(self.ax_color, 1, QtCore.Qt.DotLine))
            p.drawLine(self._last_pos.x(), area.top(), self._last_pos.x(), area.bottom())

    def _draw_title(self, p: QtGui.QPainter):
        p.setPen(self.fg_color); p.drawText(8, 16, self.title)

    def _data_bounds(self):
        xs_all = [x for s in self.series for x in s["xs"]]
        ys_all = [y for s in self.series for y in s["ys"]]
        if not xs_all or not ys_all: return 0,1,0,1
        xmin = min(xs_all) if self._xrange is None else self._xrange[0]
        xmax = max(xs_all) if self._xrange is None else self._xrange[1]
        ymin = min(ys_all) if self._yrange is None else self._yrange[0]
        ymax = max(ys_all) if self._yrange is None else self._yrange[1]
        if xmax-xmin < 1e-12: xmax = xmin + 1.0
        if ymax-ymin < 1e-12: ymax = ymin + 1.0
        return xmin,xmax,ymin,ymax

    def save_png(self, path: str):
        img = QtGui.QImage(self.size(), QtGui.QImage.Format_ARGB32); img.fill(self.bg_color)
        qp = QtGui.QPainter(img); self.render(qp); qp.end(); img.save(path)

# ---------------------------- Heatmap Görüntüleyici ------------------------ #
def color_map(value, vmin, vmax):
    if vmax <= vmin: vmax = vmin + 1e-9
    t = (value - vmin)/(vmax - vmin)
    t = max(0.0, min(1.0, t))
    r = int(0 + t*(230-0))
    g = int(70 + t*(50-70))
    b = int(200 + t*(50-200))
    return QtGui.QColor(max(0,min(255,r)),max(0,min(255,g)),max(0,min(255,b)))

class HeatmapWidget(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.values = []  # 2D list [ny][nx]
        self.vmin = 0.0; self.vmax = 1.0
        self.min_ij = None  # (i,j)
        self.setMinimumHeight(220)
        self.bg_color = QtGui.QColor(30,32,38)
        self.fg_color = QtGui.QColor(220,220,220)
        self.ax_color = QtGui.QColor(120,120,120)

    def set_theme(self, bg: QtGui.QColor, fg: QtGui.QColor):
        self.bg_color = QtGui.QColor(bg); self.fg_color = QtGui.QColor(fg)
        g = 0.5*(bg.red()+bg.green()+bg.blue())/255.0
        ax = int(90 + (100 if g<0.5 else -40))
        self.ax_color = QtGui.QColor(ax,ax,ax); self.update()

    def set_data(self, values: List[List[float]], min_ij: Optional[Tuple[int,int]]):
        self.values = values or []; self.min_ij = min_ij
        if self.values:
            flat = [v for row in self.values for v in row]
            self.vmin = min(flat); self.vmax = max(flat)
        self.update()

    def paintEvent(self, e):
        p = QtGui.QPainter(self)
        p.fillRect(self.rect(), self.bg_color)
        if not self.values:
            p.setPen(self.fg_color)
            p.drawText(self.rect(), QtCore.Qt.AlignCenter, "Heatmap yok"); return
        ny = len(self.values); nx = len(self.values[0]) if ny>0 else 0
        if nx==0: return
        margin = 40
        area = QtCore.QRect(margin, margin, self.width()-2*margin, self.height()-2*margin)
        cw = area.width()/nx; ch = area.height()/ny
        for j in range(ny):
            for i in range(nx):
                val = self.values[j][i]
                p.fillRect(QtCore.QRectF(area.left()+i*cw, area.top()+j*ch, cw+0.5, ch+0.5), color_map(val, self.vmin, self.vmax))
        if self.min_ij:
            i,j = self.min_ij
            p.setPen(QtGui.QPen(QtGui.QColor(255,255,255), 2))
            p.drawRect(QtCore.QRectF(area.left()+i*cw, area.top()+j*ch, cw, ch))
        p.setPen(self.ax_color); p.drawRect(area)
        p.setPen(self.fg_color); p.drawText(8,16,"Adsorption Lateral Heatmap (rel. E)")

    def save_png(self, path: str):
        img = QtGui.QImage(self.size(), QtGui.QImage.Format_ARGB32)
        img.fill(self.bg_color)
        qp = QtGui.QPainter(img); self.render(qp); qp.end(); img.save(path)

# ------------------------------- NEB Yardımcı ------------------------------ #
def neb_reaction_coordinate(frames: List[Frame]) -> List[float]:
    if not frames: return []
    rc = [0.0]
    for i in range(1,len(frames)):
        a, b = frames[i-1], frames[i]
        n = min(len(a.positions), len(b.positions))
        if n==0:
            step = 0.0
        else:
            s = 0.0
            for k in range(n):
                dx = a.positions[k][0]-b.positions[k][0]
                dy = a.positions[k][1]-b.positions[k][1]
                dz = a.positions[k][2]-b.positions[k][2]
                s += math.sqrt(dx*dx+dy*dy+dz*dz)
            step = s/max(1,n)
        rc.append(rc[-1] + step)
    return rc

# ------------------------------- DOS/BANDS I/O ----------------------------- #
def parse_simple_dos(path: str) -> Tuple[List[float], List[float]]:
    xs, ys = [], []
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        for ln in f:
            if not ln.strip() or ln.strip().startswith('#'): continue
            parts = ln.replace(',', ' ').replace(';',' ').split()
            if len(parts) < 2: continue
            try: x = float(parts[0]); y = float(parts[1])
            except: continue
            xs.append(x); ys.append(y)
    return xs, ys

def parse_bands_generic(path: str) -> Dict[str, List[Tuple[float,float]]]:
    bands: Dict[str, List[Tuple[float,float]]] = {}
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        for ln in f:
            if not ln.strip() or ln.strip().startswith('#'): continue
            pr = ln.replace(',', ' ').replace(';',' ').split()
            try: vals = [float(v) for v in pr]
            except: continue
            if len(vals) >= 4:
                klen = math.sqrt(vals[0]**2 + vals[1]**2 + vals[2]**2)
                E = vals[3]
                bands.setdefault('band1', []).append((klen, E))
            elif len(vals) >= 2:
                k = vals[0]
                for bi, E in enumerate(vals[1:]):
                    name = f'band{bi+1}'
                    bands.setdefault(name, []).append((k, E))
    return bands

# ------------------------------- Büyük Viewer ------------------------------ #
class BigViewer(QtWidgets.QMainWindow):
    def __init__(self, frames: List[Frame], parent=None, bg=None, fg=None):
        super().__init__(parent)
        self.setWindowTitle("Büyük Görüntüleyici")
        self.resize(1200, 800)
        self.view = MoleculeView(); self.setCentralWidget(self.view)
        if bg is not None and fg is not None:
            self._apply_palette(bg, fg)
            self.view.set_theme(bg, fg)
        self.view.set_frames(frames)
        tb = self.addToolBar("Kontrol")
        actPlay = tb.addAction("▶"); actPause = tb.addAction("⏸")
        actPrev = tb.addAction("⟨"); actNext = tb.addAction("⟩")
        actFit  = tb.addAction("Fit"); actReset= tb.addAction("Reset")
        actPlay.triggered.connect(self.view.play); actPause.triggered.connect(self.view.pause)
        actPrev.triggered.connect(self.view.prev_frame); actNext.triggered.connect(self.view.next_frame)
        actFit.triggered.connect(lambda: (self.view.fit_to_view(), self.view.update()))
        actReset.triggered.connect(lambda: (setattr(self.view, "elev", 20.0), setattr(self.view, "azim", 35.0),
                                            setattr(self.view, "_zoom", 1.0), setattr(self.view, "_pan", QtCore.QPointF(0,0)), self.view.update()))

    def _apply_palette(self, bg, fg):
        pal = QtGui.QPalette()
        pal.setColor(QtGui.QPalette.Window, bg)
        pal.setColor(QtGui.QPalette.Base, bg)
        pal.setColor(QtGui.QPalette.AlternateBase, bg)
        pal.setColor(QtGui.QPalette.Text, fg)
        pal.setColor(QtGui.QPalette.Button, bg.darker(110))
        pal.setColor(QtGui.QPalette.ButtonText, fg)
        pal.setColor(QtGui.QPalette.Highlight, QtGui.QColor(64,128,255))
        pal.setColor(QtGui.QPalette.HighlightedText, QtGui.QColor(255,255,255) if fg.lightness()<140 else QtGui.QColor(0,0,0))
        self.setPalette(pal)

# --------------------------------- Ana UI ---------------------------------- #
class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("QE Suite — Qt5 (tek dosya)")
        self.resize(1620, 1024)
        self.parser = QEParser()
        self.frames: List[Frame] = []
        self.metrics: List[StepMetrics] = []
        self.colors = palette_colors()

        # ============================ Tema kontrolleri ====================== #
        self.cb_bg = QtWidgets.QComboBox(); self.cb_bg.addItems(["Siyah", "Beyaz", "Gri"])
        self.cb_fg = QtWidgets.QComboBox(); self.cb_fg.addItems(["Beyaz", "Siyah", "Gri"])
        self.cb_bg.setCurrentIndex(0); self.cb_fg.setCurrentIndex(0)
        self.cb_bg.currentIndexChanged.connect(self.apply_theme_from_widgets)
        self.cb_fg.currentIndexChanged.connect(self.apply_theme_from_widgets)

        # ----- Dock: Animasyon -----
        self.view = MoleculeView()
        self.viewDock = QtWidgets.QDockWidget("Animasyon", self)
        self.viewDock.setWidget(self.view)
        self.viewDock.setFeatures(QtWidgets.QDockWidget.DockWidgetMovable | QtWidgets.QDockWidget.DockWidgetFloatable)
        self.addDockWidget(QtCore.Qt.RightDockWidgetArea, self.viewDock)
        self.view.requestStatus.connect(self.statusBar().showMessage)

        # ----- Sol panel -----
        left = QtWidgets.QWidget(); vleft = QtWidgets.QVBoxLayout(left)
        left.setMinimumWidth(600)

        # Dosya seçiciler
        self.le_combined = self._picker("Combined-out", open_=True)
        self.le_traj = self._picker("Traj/XYZ (ops.)", open_=True)
        self.le_surface = self._picker("Surface out", open_=True)
        self.le_adsorb = self._picker("Adsorbate out", open_=True)

        # Görünüm/çizim ayarları
        self.cb_bonds = QtWidgets.QCheckBox("Bağları çiz"); self.cb_bonds.setChecked(True)
        self.cb_cell = QtWidgets.QCheckBox("Hücre kutusunu çiz"); self.cb_cell.setChecked(True)
        self.sp_bondscale = self._dblspin(1.15, 0.5, 3.0, 0.01, "Bond cutoff scale")
        self.sp_bondwidth = self._dblspin(2.0, 1.0, 8.0, 0.2, "Bağ çizgi kalınlığı (px)")
        self.sp_atomscale = self._dblspin(1.00, 0.2, 3.0, 0.05, "Atom yarıçap x")
        self.sb_down = QtWidgets.QSpinBox(); self.sb_down.setRange(1, 50); self.sb_down.setValue(1); self.sb_down.setToolTip(">1: her n'inci atomu çiz")
        self.sb_interval = QtWidgets.QSpinBox(); self.sb_interval.setRange(20, 5000); self.sb_interval.setValue(400); self.sb_interval.setSuffix(" ms")

        # Export (GIF/MP4)
        self.sb_fps = QtWidgets.QSpinBox(); self.sb_fps.setRange(1, 60); self.sb_fps.setValue(12); self.sb_fps.setSuffix(" fps")
        self.sb_fr_a = QtWidgets.QSpinBox(); self.sb_fr_a.setRange(1, 999999); self.sb_fr_a.setValue(1)
        self.sb_fr_b = QtWidgets.QSpinBox(); self.sb_fr_b.setRange(1, 999999); self.sb_fr_b.setValue(1)
        self.sp_export_scale = self._dblspin(1.0, 0.2, 3.0, 0.05, "Çıktı ölçeği (x)")
        self.btn_export_gif = QtWidgets.QPushButton("GIF Dışa Aktar")
        self.btn_export_mp4 = QtWidgets.QPushButton("MP4 Dışa Aktar")

        # Ana butonlar
        self.btn_load = QtWidgets.QPushButton("Verileri Yükle")
        self.btn_play = QtWidgets.QPushButton("▶"); self.btn_pause = QtWidgets.QPushButton("⏸")
        self.btn_prev = QtWidgets.QPushButton("⟨"); self.btn_next = QtWidgets.QPushButton("⟩")
        self.btn_save_png = QtWidgets.QPushButton("Kareyi PNG Kaydet")
        self.btn_export_csv = QtWidgets.QPushButton("Metrikleri CSV Kaydet")
        self.btn_big = QtWidgets.QPushButton("Büyük Görüntüleyici Aç")

        # Metrik grafikleri
        self.plot_energy = MultiLinePlot(title="Total energy (eV) vs step")
        self.plot_force  = MultiLinePlot(title="Max force vs step")
        self.plot_scf    = MultiLinePlot(title="SCF accuracy (Ry) vs step")
        self.plot_mag    = MultiLinePlot(title="Total magnetization (µB) vs step")
        # PNG export butonları
        self.btn_plot_energy_png = QtWidgets.QPushButton("PNG")
        self.btn_plot_force_png  = QtWidgets.QPushButton("PNG")
        self.btn_plot_scf_png    = QtWidgets.QPushButton("PNG")
        self.btn_plot_mag_png    = QtWidgets.QPushButton("PNG")

        pg = QtWidgets.QGridLayout()
        pg.addWidget(self.plot_energy, 0,0,1,2); pg.addWidget(self.plot_force, 0,2,1,2)
        pg.addWidget(self.plot_scf,    1,0,1,2); pg.addWidget(self.plot_mag,   1,2,1,2)
        pg.addWidget(self.btn_plot_energy_png, 0,0,1,1, alignment=QtCore.Qt.AlignLeft|QtCore.Qt.AlignTop)
        pg.addWidget(self.btn_plot_force_png,  0,2,1,1, alignment=QtCore.Qt.AlignLeft|QtCore.Qt.AlignTop)
        pg.addWidget(self.btn_plot_scf_png,    1,0,1,1, alignment=QtCore.Qt.AlignLeft|QtCore.Qt.AlignTop)
        pg.addWidget(self.btn_plot_mag_png,    1,2,1,1, alignment=QtCore.Qt.AlignLeft|QtCore.Qt.AlignTop)
        plot_wrap = QtWidgets.QWidget(); plot_wrap.setLayout(pg)

        # ------------------------ Adsorpsiyon sekmesi ------------------------ #
        self.ed_ads = QtWidgets.QTextEdit(); self.ed_ads.setReadOnly(True)
        self.btn_compute_ads = QtWidgets.QPushButton("Adsorpsiyonu Hesapla")

        ads_group = QtWidgets.QGroupBox("İleri Adsorpsiyon Analizi")
        self.rb_anchor_auto = QtWidgets.QRadioButton("Anchor: En yüksek z atomu"); self.rb_anchor_auto.setChecked(True)
        self.rb_anchor_idx = QtWidgets.QRadioButton("Anchor: Elle atom indeksi")
        self.sb_anchor_idx = QtWidgets.QSpinBox(); self.sb_anchor_idx.setRange(1, 999999); self.sb_anchor_idx.setEnabled(False)
        self.rb_anchor_idx.toggled.connect(lambda b: self.sb_anchor_idx.setEnabled(b))
        self.sb_traj_N = QtWidgets.QSpinBox(); self.sb_traj_N.setRange(1, 10000); self.sb_traj_N.setValue(10); self.sb_traj_N.setSuffix(" kare")
        self.btn_ads_advanced = QtWidgets.QPushButton("İleri Analizi Çalıştır")
        agl = QtWidgets.QGridLayout(ads_group)
        agl.addWidget(self.rb_anchor_auto, 0,0,1,3)
        agl.addWidget(self.rb_anchor_idx, 1,0,1,1); agl.addWidget(QtWidgets.QLabel("Atom #:"), 1,1); agl.addWidget(self.sb_anchor_idx, 1,2)
        agl.addWidget(QtWidgets.QLabel("Son N kare:"), 2,0); agl.addWidget(self.sb_traj_N, 2,1); agl.addWidget(self.btn_ads_advanced,2,2)

        # Heatmap grubu (gelişmiş)
        heat_group = QtWidgets.QGroupBox("Adsorption Lateral Heatmap (hızlı yaklaşım)")
        self.sp_hm_nx = QtWidgets.QSpinBox(); self.sp_hm_nx.setRange(5, 200); self.sp_hm_nx.setValue(30)
        self.sp_hm_ny = QtWidgets.QSpinBox(); self.sp_hm_ny.setRange(5, 200); self.sp_hm_ny.setValue(30)
        self.sp_hm_h0 = self._dblspin(1.50, 0.2, 5.0, 0.05, "Yüzeyin üstünde sabit yükseklik (Å)")
        self.sp_hm_sigma = self._dblspin(1.00, 0.2, 3.0, 0.05, "Sigma ölçeği (yarıçap toplamına çarpan)")
        self.sp_hm_eps = self._dblspin(0.02, 0.001, 0.50, 0.001, "Epsilon (eV) — LJ benzeri potansiyel ağırlığı")
        self.btn_hm_compute = QtWidgets.QPushButton("Heatmap Hesapla")
        self.btn_hm_png = QtWidgets.QPushButton("Heatmap PNG")
        self.btn_hm_csv = QtWidgets.QPushButton("Heatmap CSV")
        self.sp_hm_h0_start = self._dblspin(0.8, 0.1, 6.0, 0.05, "h0 başlangıç (Å)")
        self.sp_hm_h0_stop  = self._dblspin(2.5, 0.1, 6.0, 0.05, "h0 bitiş (Å)")
        self.sb_hm_h0_steps = QtWidgets.QSpinBox(); self.sb_hm_h0_steps.setRange(2, 200); self.sb_hm_h0_steps.setValue(20)
        self.btn_hm_scan = QtWidgets.QPushButton("h0 Tara")
        self.btn_hm_scan_csv = QtWidgets.QPushButton("Emin(h0) CSV")
        self.heatmap = HeatmapWidget()
        self.hm_scan_plot = MultiLinePlot(title="Emin(h0) — Heatmap yüksekliği taraması")
        hg = QtWidgets.QGridLayout(heat_group)
        r=0
        hg.addWidget(QtWidgets.QLabel("Nx:"),r,0); hg.addWidget(self.sp_hm_nx,r,1)
        hg.addWidget(QtWidgets.QLabel("Ny:"),r,2); hg.addWidget(self.sp_hm_ny,r,3)
        hg.addWidget(QtWidgets.QLabel("h0 [Å]:"),r,4); hg.addWidget(self.sp_hm_h0,r,5)
        hg.addWidget(QtWidgets.QLabel("σ-scale:"),r,6); hg.addWidget(self.sp_hm_sigma,r,7)
        hg.addWidget(QtWidgets.QLabel("ε [eV]:"),r,8); hg.addWidget(self.sp_hm_eps,r,9)
        hg.addWidget(self.btn_hm_compute,r,10); hg.addWidget(self.btn_hm_png,r,11); hg.addWidget(self.btn_hm_csv,r,12)
        r+=1
        hg.addWidget(QtWidgets.QLabel("h0 başla:"),r,0); hg.addWidget(self.sp_hm_h0_start,r,1)
        hg.addWidget(QtWidgets.QLabel("h0 bitiş:"),r,2); hg.addWidget(self.sp_hm_h0_stop,r,3)
        hg.addWidget(QtWidgets.QLabel("Adım:"),r,4); hg.addWidget(self.sb_hm_h0_steps,r,5)
        hg.addWidget(self.btn_hm_scan,r,6); hg.addWidget(self.btn_hm_scan_csv,r,7); 
        r+=1
        hg.addWidget(self.heatmap, r,0,1,13); r+=1
        hg.addWidget(self.hm_scan_plot, r,0,1,13)

        ads_lay = QtWidgets.QVBoxLayout()
        ads_lay.addWidget(self.ed_ads, 1)
        ads_lay.addWidget(self.btn_compute_ads, 0, alignment=QtCore.Qt.AlignRight)
        ads_lay.addWidget(ads_group)
        ads_lay.addWidget(heat_group)
        ads_wrap = QtWidgets.QWidget(); ads_wrap.setLayout(ads_lay)

        # --------------------------- NEB sekmesi --------------------------- #
        self.neb_files = QtWidgets.QPlainTextEdit()
        self.neb_files.setPlaceholderText("NEB image .out dosyalarını her satıra bir yol şeklinde girin…")
        self.btn_neb_load = QtWidgets.QPushButton("NEB Yükle & Çiz")
        self.cb_neb_reactcoord = QtWidgets.QCheckBox("Reaksiyon Koordinatı ile çiz")
        self.neb_plot = MultiLinePlot(title="NEB Enerji Profili (ΔE, eV)")
        self.btn_neb_png = QtWidgets.QPushButton("PNG")
        neb_lay = QtWidgets.QVBoxLayout()
        neb_lay.addWidget(QtWidgets.QLabel("NEB image dosyaları:"))
        neb_lay.addWidget(self.neb_files, 1)
        hneb = QtWidgets.QHBoxLayout()
        hneb.addWidget(self.btn_neb_load); hneb.addWidget(self.cb_neb_reactcoord); hneb.addStretch(1); hneb.addWidget(self.btn_neb_png)
        neb_lay.addLayout(hneb)
        neb_lay.addWidget(self.neb_plot, 0)
        neb_wrap = QtWidgets.QWidget(); neb_wrap.setLayout(neb_lay)

        # ------------------------- DOS/PDOS sekmesi ------------------------ #
        self.dos_files = QtWidgets.QPlainTextEdit()
        self.dos_files.setPlaceholderText("DOS/PDOS dosyaları (E DOS …), her satıra bir yol…")
        self.btn_dos_load = QtWidgets.QPushButton("DOS/PDOS Çiz (çoklu)")
        self.btn_dos_png = QtWidgets.QPushButton("PNG")
        self.sp_dos_ef = self._dblspin(0.0, -100.0, 100.0, 0.01, "Fermi enerjisi (eV, integral için)")
        self.btn_dos_cum = QtWidgets.QPushButton("Cumulative (∫DOS dE)")
        self.dos_plot = MultiLinePlot(title="DOS / PDOS (çoklu)")
        dos_lay = QtWidgets.QVBoxLayout()
        dos_lay.addWidget(QtWidgets.QLabel("DOS/PDOS dosyaları:"))
        dos_lay.addWidget(self.dos_files, 1)
        hdos = QtWidgets.QHBoxLayout()
        hdos.addWidget(self.btn_dos_load); 
        hdos.addWidget(QtWidgets.QLabel("E_F:")); hdos.addWidget(self.sp_dos_ef)
        hdos.addWidget(self.btn_dos_cum)
        hdos.addStretch(1); hdos.addWidget(self.btn_dos_png)
        dos_lay.addLayout(hdos)
        dos_lay.addWidget(self.dos_plot, 0)
        dos_wrap = QtWidgets.QWidget(); dos_wrap.setLayout(dos_lay)

        # --------------------------- Bands sekmesi ------------------------- #
        self.band_files = QtWidgets.QPlainTextEdit()
        self.band_files.setPlaceholderText("Band dosyası (k_index E1 E2 … veya kx ky kz E). İlk dosyadan tüm bantlar çizilir.")
        self.btn_band_load = QtWidgets.QPushButton("Band Çiz (çok bant)")
        self.btn_band_png = QtWidgets.QPushButton("PNG")
        self.band_plot = MultiLinePlot(title="Bands")
        band_lay = QtWidgets.QVBoxLayout()
        band_lay.addWidget(QtWidgets.QLabel("Band dosyaları (en az 1 satır):"))
        band_lay.addWidget(self.band_files, 1)
        hband = QtWidgets.QHBoxLayout()
        hband.addWidget(self.btn_band_load); hband.addStretch(1); hband.addWidget(self.btn_band_png)
        band_lay.addLayout(hband)
        band_lay.addWidget(self.band_plot, 0)
        band_wrap = QtWidgets.QWidget(); band_wrap.setLayout(band_lay)

        # ---------------------------- BET sekmesi -------------------------- #
        bet_wrap = self._build_bet_tab()

        # Tabs
        tabs = QtWidgets.QTabWidget()
        tabs.addTab(plot_wrap, "Metrikler")
        tabs.addTab(ads_wrap, "Adsorpsiyon")
        tabs.addTab(neb_wrap, "NEB")
        tabs.addTab(dos_wrap, "DOS/PDOS")
        tabs.addTab(band_wrap, "Bands")
        tabs.addTab(bet_wrap, "BET (İzoterm)")

        # Sol form
        form = QtWidgets.QFormLayout()
        # Tema seçim satırı
        th = QtWidgets.QHBoxLayout()
        th.addWidget(QtWidgets.QLabel("Arka plan:")); th.addWidget(self.cb_bg)
        th.addWidget(QtWidgets.QLabel("Yazı:")); th.addWidget(self.cb_fg); th.addStretch(1)
        form.addRow(th)

        form.addRow("Combined-out:", self.le_combined)
        form.addRow("Traj/XYZ:", self.le_traj)
        form.addRow("Surface out:", self.le_surface)
        form.addRow("Adsorbate out:", self.le_adsorb)

        h1 = QtWidgets.QHBoxLayout(); h1.addWidget(self.cb_bonds); h1.addWidget(self.cb_cell)
        form.addRow(h1)

        h2 = QtWidgets.QHBoxLayout()
        h2.addWidget(QtWidgets.QLabel("Bond cutoff x:")); h2.addWidget(self.sp_bondscale)
        h2.addWidget(QtWidgets.QLabel("Bond kalınlık:")); h2.addWidget(self.sp_bondwidth)
        form.addRow(h2)

        h3 = QtWidgets.QHBoxLayout()
        h3.addWidget(QtWidgets.QLabel("Atom r. x:")); h3.addWidget(self.sp_atomscale)
        h3.addWidget(QtWidgets.QLabel("Downsample:")); h3.addWidget(self.sb_down)
        form.addRow(h3)

        form.addRow(QtWidgets.QLabel("Kare aralığı:"), self.sb_interval)

        # Export parametreleri
        exp = QtWidgets.QGridLayout()
        exp.addWidget(QtWidgets.QLabel("FPS:"), 0,0); exp.addWidget(self.sb_fps, 0,1)
        exp.addWidget(QtWidgets.QLabel("Frame aralığı: A→B"), 0,2)
        exp.addWidget(self.sb_fr_a, 0,3); exp.addWidget(self.sb_fr_b, 0,4)
        exp.addWidget(QtWidgets.QLabel("Çöz. ölçeği:"), 0,5); exp.addWidget(self.sp_export_scale, 0,6)
        exp.addWidget(self.btn_export_gif, 0,7); exp.addWidget(self.btn_export_mp4, 0,8)

        vleft.addLayout(form)
        ctrl = QtWidgets.QHBoxLayout()
        for b in (self.btn_load, self.btn_prev, self.btn_play, self.btn_pause, self.btn_next, self.btn_save_png, self.btn_export_csv, self.btn_big):
            ctrl.addWidget(b)
        vleft.addLayout(ctrl)
        vleft.addLayout(exp)
        vleft.addWidget(tabs, 1)

        central = QtWidgets.QWidget(); lay = QtWidgets.QHBoxLayout(central)
        lay.addWidget(left, 1)
        self.setCentralWidget(central)

        # ----- Sinyaller -----
        self.cb_bonds.toggled.connect(self._style_changed)
        self.cb_cell.toggled.connect(self._style_changed)
        self.sp_bondscale.valueChanged.connect(self._style_changed)
        self.sp_bondwidth.valueChanged.connect(self._style_changed)
        self.sp_atomscale.valueChanged.connect(self._style_changed)
        self.sb_down.valueChanged.connect(self._style_changed)
        self.sb_interval.valueChanged.connect(lambda v: self.view.set_interval(v))
        self.btn_load.clicked.connect(self.load_data)
        self.btn_play.clicked.connect(self.view.play); self.btn_pause.clicked.connect(self.view.pause)
        self.btn_prev.clicked.connect(self.view.prev_frame); self.btn_next.clicked.connect(self.view.next_frame)
        self.btn_save_png.clicked.connect(self.save_png)
        self.btn_export_csv.clicked.connect(self.save_csv)
        self.btn_big.clicked.connect(self.open_big_viewer)
        self.btn_compute_ads.clicked.connect(self.compute_adsorption)
        self.btn_ads_advanced.clicked.connect(self.adsorption_advanced)
        self.btn_hm_compute.clicked.connect(self.compute_heatmap)
        self.btn_hm_png.clicked.connect(self.save_heatmap_png)
        self.btn_hm_csv.clicked.connect(self.save_heatmap_csv)
        self.btn_hm_scan.clicked.connect(self.compute_heatmap_scan)
        self.btn_hm_scan_csv.clicked.connect(self.save_heatmap_scan_csv)
        self.btn_neb_load.clicked.connect(self.load_neb)
        self.btn_neb_png.clicked.connect(lambda: self.save_plot_png(self.neb_plot, "neb_plot.png"))
        self.btn_dos_load.clicked.connect(self.load_dos)
        self.btn_dos_png.clicked.connect(lambda: self.save_plot_png(self.dos_plot, "dos_plot.png"))
        self.btn_dos_cum.clicked.connect(self.dos_cumulative)
        self.btn_band_load.clicked.connect(self.load_bands)
        self.btn_band_png.clicked.connect(lambda: self.save_plot_png(self.band_plot, "band_plot.png"))

        # BET sinyalleri
        self.btn_bet_load.clicked.connect(self.bet_load_isotherm)
        self.btn_bet_compute.clicked.connect(self.bet_compute_single)
        self.btn_bet_isopng.clicked.connect(lambda: self.save_plot_png(self.bet_iso_plot, "bet_isotherm.png"))
        self.btn_bet_betpng.clicked.connect(lambda: self.save_plot_png(self.bet_plot, "bet_plot.png"))
        self.btn_bet_csv.clicked.connect(self.bet_save_csv)
        self.btn_bet_add.clicked.connect(self.bet_add_file)
        self.btn_bet_run_multi.clicked.connect(self.bet_run_multi)
        self.btn_bet_langpng.clicked.connect(lambda: self.save_plot_png(self.bet_lang_plot, "bet_langmuir.png"))
        self.btn_bet_tpng.clicked.connect(lambda: self.save_plot_png(self.bet_t_plot, "bet_tplot.png"))

        self.btn_export_gif.clicked.connect(lambda: self.export_animation("gif"))
        self.btn_export_mp4.clicked.connect(lambda: self.export_animation("mp4"))

        self.status = self.statusBar()

        # Ayarlar + tema
        self.load_settings()
        self.apply_theme_from_widgets()

    # ----------------------- BET TAB inşası -------------------------------- #
    def _build_bet_tab(self):
        wrap = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(wrap)

        self.le_bet = self._picker("İzoterm dosyası (P/P0, V)", open_=True)
        self.btn_bet_load = QtWidgets.QPushButton("İzoterm Yükle")
        self.btn_bet_compute = QtWidgets.QPushButton("BET Hesapla")
        self.btn_bet_csv = QtWidgets.QPushButton("BET CSV")

        # Üniteler
        unit_group = QtWidgets.QGroupBox("V birimi")
        self.rb_bet_cm3 = QtWidgets.QRadioButton("cm³(STP)/g"); self.rb_bet_cm3.setChecked(True)
        self.rb_bet_mol = QtWidgets.QRadioButton("mol/g")
        uh = QtWidgets.QHBoxLayout(unit_group)
        uh.addWidget(self.rb_bet_cm3); uh.addWidget(self.rb_bet_mol)

        # Parametreler
        self.sp_bet_xmin = self._dblspin(0.05, 0.000, 0.95, 0.01, "BET fit alt sınır P/P0")
        self.sp_bet_xmax = self._dblspin(0.35, 0.001, 0.999, 0.01, "BET fit üst sınır P/P0")
        self.sp_bet_sigma = self._dblspin(0.162, 0.010, 0.400, 0.001, "Molekül kesit alanı σ (nm², N2/77K için 0.162)")
        self.sp_bet_vmolar = self._dblspin(22414.0, 10000.0, 26000.0, 1.0, "STP molar hacim (cm³/mol)")
        self.sp_bet_T = self._dblspin(77.0, 20.0, 500.0, 0.1, "Sıcaklık T (K)")
        self.btn_bet_isopng = QtWidgets.QPushButton("Isotherm PNG")
        self.btn_bet_betpng = QtWidgets.QPushButton("BET PNG")

        # Çoklu izoterm
        self.bet_multi = QtWidgets.QPlainTextEdit()
        self.bet_multi.setPlaceholderText("Çoklu izoterm için dosya yollarını her satıra bir tane yazın (opsiyonel etiket için \",\" ile ekleyin).")
        self.btn_bet_add = QtWidgets.QPushButton("Ekle…")
        self.btn_bet_run_multi = QtWidgets.QPushButton("Çoklu Analiz (BET+Langmuir+t-plot)")

        # Langmuir & t-plot parametreleri
        self.sp_L_xmin = self._dblspin(0.02, 0.000, 0.95, 0.01, "Langmuir alt sınır P/P0")
        self.sp_L_xmax = self._dblspin(0.25, 0.001, 0.999, 0.01, "Langmuir üst sınır P/P0")
        self.cb_t_model = QtWidgets.QComboBox(); self.cb_t_model.addItems(["Harkins–Jura", "Halsey"])
        self.sp_t_xmin = self._dblspin(0.35, 0.000, 0.95, 0.01, "t-plot alt sınır P/P0")
        self.sp_t_xmax = self._dblspin(0.80, 0.001, 0.999, 0.01, "t-plot üst sınır P/P0")

        grid = QtWidgets.QGridLayout()
        grid.addWidget(QtWidgets.QLabel("P/P0 alt:"),0,0); grid.addWidget(self.sp_bet_xmin,0,1)
        grid.addWidget(QtWidgets.QLabel("P/P0 üst:"),0,2); grid.addWidget(self.sp_bet_xmax,0,3)
        grid.addWidget(QtWidgets.QLabel("σ [nm²]:"),1,0); grid.addWidget(self.sp_bet_sigma,1,1)
        grid.addWidget(QtWidgets.QLabel("V̄m [cm³/mol]:"),1,2); grid.addWidget(self.sp_bet_vmolar,1,3)
        grid.addWidget(QtWidgets.QLabel("T [K]:"),1,4); grid.addWidget(self.sp_bet_T,1,5)
        grid.addWidget(self.btn_bet_compute,0,4); grid.addWidget(self.btn_bet_csv,0,5)

        # Grafikler
        self.bet_iso_plot = MultiLinePlot(title="İzoterm: V vs P/P0")
        self.bet_plot = MultiLinePlot(title="BET Plot: Y = 1/[V((1/x)−1)] vs x")
        self.bet_lang_plot = MultiLinePlot(title="Langmuir Lineerleştirme: x/V vs x")
        self.bet_t_plot = MultiLinePlot(title="t-plot: V vs t (nm)")
        self.bet_summary = QtWidgets.QTextEdit(); self.bet_summary.setReadOnly(True)

        # üst satır
        top = QtWidgets.QHBoxLayout()
        top.addWidget(self.le_bet, 1); top.addWidget(self.btn_bet_load); 
        top.addWidget(unit_group); top.addStretch(1); 
        top.addWidget(self.btn_bet_isopng); top.addWidget(self.btn_bet_betpng)

        # ikinci satır: Langmuir & t-plot param
        top2 = QtWidgets.QHBoxLayout()
        top2.addWidget(QtWidgets.QLabel("Langmuir P/P0:")); top2.addWidget(self.sp_L_xmin); top2.addWidget(QtWidgets.QLabel("→")); top2.addWidget(self.sp_L_xmax)
        top2.addSpacing(20)
        top2.addWidget(QtWidgets.QLabel("t-plot model:")); top2.addWidget(self.cb_t_model)
        top2.addWidget(QtWidgets.QLabel(" P/P0:")); top2.addWidget(self.sp_t_xmin); top2.addWidget(QtWidgets.QLabel("→")); top2.addWidget(self.sp_t_xmax)
        top2.addStretch(1)
        self.btn_bet_langpng = QtWidgets.QPushButton("Langmuir PNG")
        self.btn_bet_tpng = QtWidgets.QPushButton("t-plot PNG")
        top2.addWidget(self.btn_bet_langpng); top2.addWidget(self.btn_bet_tpng)

        # üçüncü satır: çoklu
        top3 = QtWidgets.QHBoxLayout()
        top3.addWidget(QtWidgets.QLabel("Çoklu izoterm listesi:"))
        top3.addStretch(1); top3.addWidget(self.btn_bet_add); top3.addWidget(self.btn_bet_run_multi)

        lay.addLayout(top)
        lay.addLayout(grid)
        lay.addLayout(top2)
        lay.addLayout(top3)
        lay.addWidget(self.bet_multi, 1)
        lay.addWidget(self.bet_iso_plot, 1)
        lay.addWidget(self.bet_plot, 1)
        lay.addWidget(self.bet_lang_plot, 1)
        lay.addWidget(self.bet_t_plot, 1)
        lay.addWidget(self.bet_summary, 1)
        return wrap

    # ----------------------- Ayarlar (QSettings) --------------------------- #
    def load_settings(self):
        st = QtCore.QSettings("QESuite", "Qt5")
        self.le_combined.lineedit.setText(st.value("combined", "", type=str))
        self.le_traj.lineedit.setText(st.value("traj", "", type=str))
        self.le_surface.lineedit.setText(st.value("surf", "", type=str))
        self.le_adsorb.lineedit.setText(st.value("adsorb", "", type=str))
        self.cb_bonds.setChecked(st.value("draw_bonds", True, type=bool))
        self.cb_cell.setChecked(st.value("draw_cell", True, type=bool))
        self.sp_bondscale.setValue(float(st.value("bond_scale", 1.15)))
        self.sp_bondwidth.setValue(float(st.value("bond_width", 2.0)))
        self.sp_atomscale.setValue(float(st.value("atom_scale", 1.0)))
        self.sb_down.setValue(int(st.value("downsample", 1)))
        self.sb_interval.setValue(int(st.value("interval", 400)))
        self.sb_fps.setValue(int(st.value("fps", 12)))
        self.sp_export_scale.setValue(float(st.value("export_scale", 1.0)))
        self.sp_hm_h0.setValue(float(st.value("hm_h0", 1.5)))
        self.sp_hm_sigma.setValue(float(st.value("hm_sigma", 1.0)))
        self.sp_hm_eps.setValue(float(st.value("hm_eps", 0.02)))
        self.sp_hm_h0_start.setValue(float(st.value("hm_h0_start", 0.8)))
        self.sp_hm_h0_stop.setValue(float(st.value("hm_h0_stop", 2.5)))
        self.sb_hm_h0_steps.setValue(int(st.value("hm_h0_steps", 20)))
        self.sp_dos_ef.setValue(float(st.value("dos_ef", 0.0)))
        self.sp_bet_xmin.setValue(float(st.value("bet_xmin", 0.05)))
        self.sp_bet_xmax.setValue(float(st.value("bet_xmax", 0.35)))
        self.sp_bet_sigma.setValue(float(st.value("bet_sigma", 0.162)))
        self.sp_bet_vmolar.setValue(float(st.value("bet_vmolar", 22414.0)))
        self.sp_bet_T.setValue(float(st.value("bet_T", 77.0)))
        self.cb_bg.setCurrentIndex(int(st.value("theme_bg_idx", 0)))
        self.cb_fg.setCurrentIndex(int(st.value("theme_fg_idx", 0)))
        if st.value("bet_unit", "cm3") == "mol":
            self.rb_bet_mol.setChecked(True)
        else:
            self.rb_bet_cm3.setChecked(True)
        self.sp_L_xmin.setValue(float(st.value("L_xmin", 0.02)))
        self.sp_L_xmax.setValue(float(st.value("L_xmax", 0.25)))
        self.cb_t_model.setCurrentIndex(int(st.value("t_model_idx", 0)))
        self.sp_t_xmin.setValue(float(st.value("t_xmin", 0.35)))
        self.sp_t_xmax.setValue(float(st.value("t_xmax", 0.80)))

    def save_settings(self):
        st = QtCore.QSettings("QESuite", "Qt5")
        st.setValue("combined", self.le_combined.lineedit.text())
        st.setValue("traj", self.le_traj.lineedit.text())
        st.setValue("surf", self.le_surface.lineedit.text())
        st.setValue("adsorb", self.le_adsorb.lineedit.text())
        st.setValue("draw_bonds", self.cb_bonds.isChecked())
        st.setValue("draw_cell", self.cb_cell.isChecked())
        st.setValue("bond_scale", self.sp_bondscale.value())
        st.setValue("bond_width", self.sp_bondwidth.value())
        st.setValue("atom_scale", self.sp_atomscale.value())
        st.setValue("downsample", self.sb_down.value())
        st.setValue("interval", self.sb_interval.value())
        st.setValue("fps", self.sb_fps.value())
        st.setValue("export_scale", self.sp_export_scale.value())
        st.setValue("hm_h0", self.sp_hm_h0.value())
        st.setValue("hm_sigma", self.sp_hm_sigma.value())
        st.setValue("hm_eps", self.sp_hm_eps.value())
        st.setValue("hm_h0_start", self.sp_hm_h0_start.value())
        st.setValue("hm_h0_stop", self.sp_hm_h0_stop.value())
        st.setValue("hm_h0_steps", self.sb_hm_h0_steps.value())
        st.setValue("dos_ef", self.sp_dos_ef.value())
        st.setValue("bet_xmin", self.sp_bet_xmin.value())
        st.setValue("bet_xmax", self.sp_bet_xmax.value())
        st.setValue("bet_sigma", self.sp_bet_sigma.value())
        st.setValue("bet_vmolar", self.sp_bet_vmolar.value())
        st.setValue("bet_T", self.sp_bet_T.value())
        st.setValue("bet_unit", "mol" if self.rb_bet_mol.isChecked() else "cm3")
        st.setValue("theme_bg_idx", self.cb_bg.currentIndex())
        st.setValue("theme_fg_idx", self.cb_fg.currentIndex())
        st.setValue("L_xmin", self.sp_L_xmin.value())
        st.setValue("L_xmax", self.sp_L_xmax.value())
        st.setValue("t_model_idx", self.cb_t_model.currentIndex())
        st.setValue("t_xmin", self.sp_t_xmin.value())
        st.setValue("t_xmax", self.sp_t_xmax.value())

    def closeEvent(self, e: QtGui.QCloseEvent):
        self.save_settings()
        super().closeEvent(e)

    # --------------------------- Yardımcı UI -------------------------------- #
    def _picker(self, placeholder: str, open_=True) -> QtWidgets.QWidget:
        le = QtWidgets.QLineEdit(); le.setPlaceholderText(placeholder)
        btn = QtWidgets.QPushButton("…")
        lay = QtWidgets.QHBoxLayout(); lay.setContentsMargins(0,0,0,0)
        wrap = QtWidgets.QWidget(); wrap.setLayout(lay)
        lay.addWidget(le, 1); lay.addWidget(btn)
        def browse():
            d = os.path.dirname(le.text()) if le.text() else os.getcwd()
            if open_:
                path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Dosya seç", d, "Tüm dosyalar (*.*)")
            else:
                path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Kaydet", d, "Tüm dosyalar (*.*)")
            if path: le.setText(path)
        btn.clicked.connect(browse)
        wrap.lineedit = le
        return wrap

    def _dblspin(self, val, mn, mx, step, tooltip):
        sp = QtWidgets.QDoubleSpinBox(); sp.setDecimals(3); sp.setRange(mn,mx); sp.setSingleStep(step); sp.setValue(val); sp.setToolTip(tooltip); return sp

    # ---------------------------- Tema uygulama ----------------------------- #
    def apply_theme_from_widgets(self):
        bg = color_from_choice(self.cb_bg.currentText())
        fg = text_from_choice(self.cb_fg.currentText())
        # aynı renk seçilmişse düzelt
        if (bg.red(),bg.green(),bg.blue()) == (fg.red(),fg.green(),fg.blue()):
            fg = QtGui.QColor(235,235,235) if bg.lightness()<128 else QtGui.QColor(10,10,10)
        self.apply_theme(bg, fg)

    def apply_theme(self, bg: QtGui.QColor, fg: QtGui.QColor):
        pal = QtGui.QPalette()
        pal.setColor(QtGui.QPalette.Window, bg)
        pal.setColor(QtGui.QPalette.Base, bg)
        pal.setColor(QtGui.QPalette.AlternateBase, bg)
        pal.setColor(QtGui.QPalette.Text, fg)
        pal.setColor(QtGui.QPalette.Button, bg.darker(110))
        pal.setColor(QtGui.QPalette.ButtonText, fg)
        pal.setColor(QtGui.QPalette.Highlight, QtGui.QColor(64,128,255))
        pal.setColor(QtGui.QPalette.HighlightedText, QtGui.QColor(255,255,255) if fg.lightness()<140 else QtGui.QColor(0,0,0))
        self.setPalette(pal)

        # özel widgetlara geç
        for w in [self.view,
                  self.plot_energy, self.plot_force, self.plot_scf, self.plot_mag,
                  self.dos_plot, self.band_plot, self.neb_plot,
                  self.bet_iso_plot, self.bet_plot, self.bet_lang_plot, self.bet_t_plot,
                  self.heatmap]:
            if hasattr(w, "set_theme"): w.set_theme(bg, fg)
        self.update()

    # ---------------------------- Stil güncelle ----------------------------- #
    def _style_changed(self, *args):
        self.view.draw_bonds = self.cb_bonds.isChecked()
        self.view.draw_cell = self.cb_cell.isChecked()
        self.view.bond_scale = float(self.sp_bondscale.value())
        self.view.bond_width = float(self.sp_bondwidth.value())
        self.view.atom_radius_scale = float(self.sp_atomscale.value())
        self.view.downsample = int(self.sb_down.value())
        self.view.update()

    # ------------------------------ Yükleme --------------------------------- #
    def load_data(self):
        self.frames, self.metrics = [], []
        combined = self.le_combined.lineedit.text().strip()
        traj = self.le_traj.lineedit.text().strip()
        try:
            if combined and os.path.isfile(combined):
                f, m, _ = self.parser.parse_file(combined)
                self.frames = f; self.metrics = m
            if (not self.frames) and traj and os.path.isfile(traj):
                f2, _, _ = self.parser.parse_file(traj)
                self.frames = f2
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Ayrıştırma Hatası", str(e)); return

        if not self.frames:
            QtWidgets.QMessageBox.warning(self, "Uyarı", "Kare bulunamadı. Dosyayı kontrol edin.")
        self.view.set_frames(self.frames)
        self.view.set_interval(self.sb_interval.value())
        self._refresh_plots()

        n = max(1, len(self.frames))
        self.sb_fr_a.setMaximum(n); self.sb_fr_b.setMaximum(n)
        self.sb_fr_a.setValue(1); self.sb_fr_b.setValue(n)

        self.status.showMessage(f"{len(self.frames)} kare, {len(self.metrics)} metrik adımı yüklendi.", 5000)

    # --------------------------- Grafikler ---------------------------------- #
    def _refresh_plots(self):
        xs = [m.step for m in self.metrics]
        self.plot_energy.set_single_series(xs, [m.total_energy_eV for m in self.metrics], QtGui.QColor(0,200,255), "E")
        self.plot_force.set_single_series(xs, [m.max_force for m in self.metrics], QtGui.QColor(255,140,0), "Fmax")
        self.plot_scf.set_single_series(xs, [m.scf_acc_Ry for m in self.metrics], QtGui.QColor(120,220,120), "scf")
        self.plot_mag.set_single_series(xs, [m.magnetization for m in self.metrics], QtGui.QColor(200,120,220), "mag")

    # ------------------------------ Çıktılar -------------------------------- #
    def save_png(self):
        if not self.frames:
            QtWidgets.QMessageBox.information(self, "Bilgi", "Önce veri yükleyin."); return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "PNG kaydet", os.getcwd(), "PNG (*.png)")
        if not path: return
        try:
            self.view.save_current_png(path); self.status.showMessage(f"Kaydedildi: {path}", 3000)
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Hata", str(e))

    def save_csv(self):
        if not self.metrics:
            QtWidgets.QMessageBox.information(self, "Bilgi", "CSV için metrik yok."); return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "CSV kaydet", os.getcwd(), "CSV (*.csv)")
        if not path: return
        try:
            with open(path, 'w', newline='') as f:
                w = csv.writer(f)
                w.writerow(["step","total_energy_eV","max_force","scf_acc_Ry","fermi_eV","magnetization"])
                for m in self.metrics:
                    w.writerow([m.step, m.total_energy_eV, m.max_force, m.scf_acc_Ry, m.fermi_eV, m.magnetization])
            self.status.showMessage(f"Kaydedildi: {path}", 3000)
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Hata", str(e))

    def save_plot_png(self, plot_widget: MultiLinePlot, default_name: str):
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "PNG kaydet", os.path.join(os.getcwd(), default_name), "PNG (*.png)")
        if not path: return
        plot_widget.save_png(path)
        self.status.showMessage(f"Kaydedildi: {path}", 3000)

    def save_heatmap_png(self):
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Heatmap PNG kaydet", os.path.join(os.getcwd(), "heatmap.png"), "PNG (*.png)")
        if not path: return
        self.heatmap.save_png(path); self.status.showMessage(f"Kaydedildi: {path}", 3000)

    # -------------------------- Adsorpsiyon (temel) ------------------------ #
    def compute_adsorption(self):
        combined = self.le_combined.lineedit.text().strip()
        surface = self.le_surface.lineedit.text().strip()
        adsorb = self.le_adsorb.lineedit.text().strip()
        Eads, Esys, Esurf, Ead = self._adsorption_energy(combined or None, surface or None, adsorb or None)
        txt = []
        def fmt(x, p=6): return f"{x:.{p}f}" if isinstance(x,(int,float)) and x is not None else "N/A"
        txt.append(f"<b>Adsorption</b>")
        txt.append(f"E_system (eV): {fmt(Esys)}")
        txt.append(f"E_surface (eV): {fmt(Esurf)}")
        txt.append(f"E_adsorbate (eV): {fmt(Ead)}")
        txt.append(f"<b>E_ads (eV): {fmt(Eads)}</b>")

        if self.frames:
            fr = self.frames[-1]
            site, anchor_sym, dmin = self.classify_site(fr)
            area = cell_base_area(fr.cell)
            txt.append("<hr>")
            txt.append(f"Site (kaba): {site or 'N/A'}")
            txt.append(f"Anchor atom (tür): {anchor_sym or 'N/A'}")
            txt.append(f"Min yüzey mesafesi (Å): {fmt(dmin,3)}")
            if Eads is not None and area:
                txt.append(f"E_ads/alan (eV/Å²): {fmt(Eads/area,4)}")

        try:
            base_sys = os.path.dirname(combined) if combined else ""
            acf_sys = self._parse_bader_acf_sum(os.path.join(base_sys, "ACF.dat")) if base_sys else None
            base_surf = os.path.dirname(surface) if surface else ""
            acf_surf = self._parse_bader_acf_sum(os.path.join(base_surf, "ACF.dat")) if base_surf else None
            base_ads = os.path.dirname(adsorb) if adsorb else ""
            acf_ads = self._parse_bader_acf_sum(os.path.join(base_ads, "ACF.dat")) if base_ads else None
            if acf_sys is not None and acf_surf is not None and acf_ads is not None:
                dq = acf_sys - (acf_surf + acf_ads)
                txt.append(f"ΔQ (Bader, e): {fmt(dq,3)}")
        except Exception:
            pass

        self.ed_ads.setHtml("<br>".join(txt))

    # -------------------- Adsorpsiyon (ileri metrikler) -------------------- #
    def adsorption_advanced(self):
        if not self.frames:
            QtWidgets.QMessageBox.information(self, "Bilgi", "Önce kare yükleyin."); return
        fr = self.frames[-1]
        anchor_idx = None
        if self.rb_anchor_idx.isChecked():
            idx = self.sb_anchor_idx.value()-1
            if 0 <= idx < len(fr.positions):
                anchor_idx = idx
        stats = adsorption_geometry_stats(fr, anchor_idx=anchor_idx)
        if not stats:
            QtWidgets.QMessageBox.information(self, "Bilgi", "Geometri istatistiği hesaplanamadı."); return

        N = min(self.sb_traj_N.value(), len(self.frames))
        hs, cns = [], []
        for frx in self.frames[-N:]:
            stx = adsorption_geometry_stats(frx, anchor_idx=stats["anchor_idx"])
            if stx:
                hs.append(stx["h"]); cns.append(stx["coordination"])
        def ssum(vs): 
            return (min(vs), sum(vs)/len(vs), max(vs)) if vs else (None,None,None)
        hmin, havg, hmax = ssum(hs); cnmin, cnavg, cnmax = ssum(cns)

        def fmt(x, p=4): return f"{x:.{p}f}" if isinstance(x,(int,float)) else "N/A"

        lines = []
        lines.append("<b>İleri Adsorpsiyon Analizi</b>")
        lines.append(f"Anchor atom indeksi: {stats['anchor_idx']+1}")
        lines.append(f"h (son kare) [Å]: {fmt(stats['h'])}")
        lines.append(f"Koordinasyon sayısı (son kare): {stats['coordination']}")
        lines.append(f"Ortalama bağ uzaklığı (son kare) [Å]: {fmt(stats['avg_bond'])}")
        if stats["nearest"]:
            lst = ", ".join([f"#{j+1}:{d:.2f}" for j,d in stats["nearest"]])
            lines.append(f"En yakın yüzey komşuları (indeks:mesafe): {lst}")
        lines.append("<hr>")
        lines.append(f"h (son {N} kare) [min/ort/max] [Å]: {fmt(hmin)} / {fmt(havg)} / {fmt(hmax)}")
        lines.append(f"Koordinasyon (son {N} kare) [min/ort/max]: {cnmin} / {fmt(cnavg,2)} / {cnmax}")
        self.ed_ads.append("\n" + "\n".join(lines))

    # ---------------------- Heatmap çekirdeği + API ------------------------ #
    def _heatmap_grid(self, fr, nx, ny, h0, sig_scale, eps):
        if not fr.positions:
            return None, None, None
        if fr.cell:
            a1, a2 = fr.cell[0], fr.cell[1]
        else:
            (xmin,xmax),(ymin,ymax),_ = bbox3(fr.positions)
            a1 = (xmax - xmin, 0.0, 0.0); a2 = (0.0, ymax - ymin, 0.0)
        surf_idx = top_layer_indices(fr)
        if not surf_idx:
            return None, None, None
        zsurf_mean = sum(fr.positions[i][2] for i in surf_idx)/len(surf_idx)
        anchor_idx = max(range(len(fr.positions)), key=lambda i: fr.positions[i][2])
        anchor_sym = fr.symbols[anchor_idx]
        sigma_anchor = cov_rad(anchor_sym)
        vals = [[0.0 for _ in range(nx)] for _ in range(ny)]
        allE = []
        for j in range(ny):
            for i in range(nx):
                s = (i + 0.5)/nx; t = (j + 0.5)/ny
                x = s*a1[0] + t*a2[0]
                y = s*a1[1] + t*a2[1]
                z = zsurf_mean + h0
                E = 0.0
                for k in surf_idx:
                    rk = cov_rad(fr.symbols[k])
                    sigma = (sigma_anchor + rk)*sig_scale
                    dx = x - fr.positions[k][0]
                    dy = y - fr.positions[k][1]
                    dz = z - fr.positions[k][2]
                    r = math.sqrt(dx*dx+dy*dy+dz*dz) + 1e-9
                    sr = sigma / r; sr6 = sr**6; sr12 = sr6*sr6
                    E += 4.0*eps*(sr12 - sr6)
                vals[j][i] = E
                allE.append(E)
        Emin = min(allE)
        for j in range(ny):
            for i in range(nx):
                vals[j][i] = vals[j][i] - Emin
        min_ij = None; mv = 1e99
        for j in range(ny):
            for i in range(nx):
                if vals[j][i] < mv: mv = vals[j][i]; min_ij = (i,j)
        return vals, min_ij, Emin

    def compute_heatmap(self):
        if not self.frames:
            QtWidgets.QMessageBox.information(self, "Bilgi", "Önce kare yükleyin."); return
        fr = self.frames[-1]
        nx = self.sp_hm_nx.value(); ny = self.sp_hm_ny.value()
        h0 = float(self.sp_hm_h0.value())
        sig_scale = float(self.sp_hm_sigma.value())
        eps = float(self.sp_hm_eps.value())
        vals, min_ij, Emin = self._heatmap_grid(fr, nx, ny, h0, sig_scale, eps)
        if vals is None:
            QtWidgets.QMessageBox.information(self, "Bilgi", "Heatmap hesaplanamadı."); return
        self._last_heatmap_vals = vals
        self._last_heatmap_min = (min_ij, Emin, h0, nx, ny, sig_scale, eps)
        self.heatmap.set_data(vals, min_ij)
        self.status.showMessage(f"Heatmap hazır (h0={h0:.2f} Å) | Emin ≈ {Emin:.4f} eV (mutlak)", 6000)

    def save_heatmap_csv(self):
        vals = getattr(self, "_last_heatmap_vals", None)
        meta = getattr(self, "_last_heatmap_min", None)
        if not vals or not meta:
            QtWidgets.QMessageBox.information(self, "Bilgi", "Önce heatmap hesaplayın."); return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Heatmap CSV kaydet", os.path.join(os.getcwd(),"heatmap.csv"), "CSV (*.csv)")
        if not path: return
        (min_ij, Emin, h0, nx, ny, sig_scale, eps) = meta
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["# h0[Å]", h0])
            w.writerow(["# nx", nx, "ny", ny, "sigma_scale", sig_scale, "eps[eV]", eps])
            w.writerow(["# Emin_abs[eV]", f"{Emin:.6f}", "min_ij", min_ij[0], min_ij[1]])
            for row in vals:
                w.writerow([f"{v:.8f}" for v in row])
        self.status.showMessage(f"Heatmap CSV kaydedildi: {path}", 5000)

    def compute_heatmap_scan(self):
        if not self.frames:
            QtWidgets.QMessageBox.information(self, "Bilgi", "Önce kare yükleyin."); return
        fr = self.frames[-1]
        nx = self.sp_hm_nx.value(); ny = self.sp_hm_ny.value()
        h0_a = float(self.sp_hm_h0_start.value())
        h0_b = float(self.sp_hm_h0_stop.value())
        steps = max(2, int(self.sb_hm_h0_steps.value()))
        sig_scale = float(self.sp_hm_sigma.value())
        eps = float(self.sp_hm_eps.value())
        if h0_b < h0_a: h0_a, h0_b = h0_b, h0_a
        hs = [h0_a + (h0_b-h0_a)*k/(steps-1) for k in range(steps)]
        Emin_list = []
        for h in hs:
            vals, min_ij, Emin = self._heatmap_grid(fr, nx, ny, h, sig_scale, eps)
            if Emin is None: Emin = float('nan')
            Emin_list.append(Emin)
        series = [{"xs":hs, "ys":Emin_list, "color": QtGui.QColor(255,200,30), "label":"Emin(h0)"}]
        self.hm_scan_plot.set_series_list(series)
        self.hm_scan_plot.set_markers([])
        self._hm_scan_cache = (hs, Emin_list)
        self.status.showMessage(f"h0 taraması tamam: {steps} nokta, h0 ∈ [{h0_a:.2f}, {h0_b:.2f}] Å", 6000)

    def save_heatmap_scan_csv(self):
        cache = getattr(self, "_hm_scan_cache", None)
        if not cache:
            QtWidgets.QMessageBox.information(self, "Bilgi", "Önce h0 taramasını çalıştırın."); return
        hs, Emin_list = cache
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Emin(h0) CSV kaydet", os.path.join(os.getcwd(),"heatmap_h0_scan.csv"), "CSV (*.csv)")
        if not path: return
        with open(path, "w", newline="") as f:
            w = csv.writer(f); w.writerow(["h0[Å]", "Emin_abs[eV]"])
            for h, e in zip(hs, Emin_list): w.writerow([f"{h:.6f}", f"{e:.8f}"])
        self.status.showMessage(f"Emin(h0) CSV kaydedildi: {path}", 5000)

    # ------------------------------- NEB ------------------------------------ #
    def load_neb(self):
        paths = [ln.strip() for ln in self.neb_files.toPlainText().splitlines() if ln.strip()]
        if not paths:
            QtWidgets.QMessageBox.information(self, "Bilgi", "NEB için dosya listesi boş."); return
        frames, energies = self._parse_neb_images(paths)
        if frames: self.view.set_frames(frames)
        xs = neb_reaction_coordinate(frames) if self.cb_neb_reactcoord.isChecked() else list(range(1, len(energies)+1))
        if any(isinstance(e, float) and not (e is None or math.isnan(e)) for e in energies):
            Emin = min(e for e in energies if e is not None and not math.isnan(e))
            en = [ (e - Emin) if (e is not None and not math.isnan(e)) else None for e in energies ]
        else:
            en = [None for _ in energies]
        sp_idx = None
        if any(v is not None for v in en):
            vmax = -1e99; sp_idx = 0
            for i,v in enumerate(en):
                if v is not None and v > vmax: vmax = v; sp_idx = i
        self.neb_plot.set_single_series(xs, en, QtGui.QColor(255,200,30), "ΔE")
        markers = []
        if sp_idx is not None and sp_idx < len(xs) and en[sp_idx] is not None:
            markers.append((xs[sp_idx], f"saddle ~ {en[sp_idx]:.3f} eV", QtGui.QColor(255,120,120)))
        self.neb_plot.set_markers(markers)
        self.status.showMessage(f"NEB: {len(frames)} image, enerji noktası: {len(en)}", 5000)

    # ------------------------------ DOS/PDOS -------------------------------- #
    def load_dos(self):
        paths = [ln.strip() for ln in self.dos_files.toPlainText().splitlines() if ln.strip()]
        if not paths:
            QtWidgets.QMessageBox.information(self, "Bilgi", "DOS/PDOS için dosya listesi boş."); return
        series = []
        for idx, p in enumerate(paths):
            if not os.path.isfile(p): continue
            xs, ys = parse_simple_dos(p)
            series.append({"xs": xs, "ys": ys, "color": self.colors[idx % len(self.colors)], "label": os.path.basename(p)})
        if not series:
            QtWidgets.QMessageBox.information(self, "Bilgi", "Hiç dosya okunamadı."); return
        for s in series:
            z = list(zip(s["xs"], s["ys"])); z.sort(key=lambda t:t[0])
            s["xs"] = [a for a,_ in z]; s["ys"] = [b for _,b in z]
        self.dos_plot.set_series_list(series)
        self.dos_last_series = series
        self.status.showMessage(f"DOS/PDOS: {len(series)} seri çizildi.", 5000)

    def dos_cumulative(self):
        series = getattr(self, "dos_last_series", None)
        if not series:
            QtWidgets.QMessageBox.information(self, "Bilgi", "Önce DOS/PDOS dosyalarını yükleyin."); return
        Ef = float(self.sp_dos_ef.value())
        ser_out = []
        for idx, s in enumerate(series):
            xs, ys = s["xs"], s["ys"]
            if not xs or len(xs)!=len(ys): continue
            z = list(zip(xs, ys)); z.sort(key=lambda t:t[0])
            xs = [a for a,_ in z]; ys = [b for _,b in z]
            cum = [0.0]
            for i in range(1,len(xs)):
                dx = xs[i]-xs[i-1]
                area = 0.5*(ys[i]+ys[i-1])*dx
                cum.append(cum[-1] + area)
            j = bisect.bisect_left(xs, Ef)
            if j<=0: Ne = cum[0]
            elif j>=len(xs): Ne = cum[-1]
            else:
                x0,x1 = xs[j-1], xs[j]; c0,c1 = cum[j-1], cum[j]
                t = (Ef-x0)/max(1e-12,(x1-x0))
                Ne = c0 + t*(c1-c0)
            col = QtGui.QColor(s["color"]); col.setAlpha(200)
            ser_out.append({"xs": xs, "ys": cum, "color": col, "label": s["label"]+" (cum)"})
            self.status.showMessage(f"{s['label']}: N(E≤{Ef:.3f} eV) ≈ {Ne:.4f} (keyfi birim)", 6000)
        ser_all = []; ser_all.extend(series); ser_all.extend(ser_out)
        self.dos_plot.set_series_list(ser_all)

    # -------------------------------- Bands --------------------------------- #
    def load_bands(self):
        paths = [ln.strip() for ln in self.band_files.toPlainText().splitlines() if ln.strip()]
        if not paths:
            QtWidgets.QMessageBox.information(self, "Bilgi", "Band için dosya listesi boş."); return
        p0 = paths[0]
        if not os.path.isfile(p0):
            QtWidgets.QMessageBox.information(self, "Bilgi", "Dosya bulunamadı."); return
        bands_all = parse_bands_generic(p0)
        if not bands_all:
            QtWidgets.QMessageBox.information(self, "Bilgi", "Band verisi okunamadı."); return
        names = sorted(bands_all.keys(), key=lambda x: int(x[4:]) if x.startswith("band") and x[4:].isdigit() else 9999)
        series = []
        for idx, name in enumerate(names):
            pts = bands_all[name]
            xs = [k for (k,_) in pts]; ys = [E for (_,E) in pts]
            z = list(zip(xs, ys)); z.sort(key=lambda t:t[0])
            xs = [a for a,_ in z]; ys = [b for _,b in z]
            series.append({"xs":xs, "ys":ys, "color": self.colors[idx % len(self.colors)], "label": name})
        self.band_plot.set_series_list(series)
        allE = [y for s in series for y in s["ys"]]
        vb = max([e for e in allE if e<=0], default=None)
        cb = min([e for e in allE if e>0], default=None)
        markers = []
        if vb is not None:
            markers.append((series[0]["xs"][0], f"VB max ~ {vb:.3f} eV", QtGui.QColor(160,200,255)))
        if cb is not None:
            markers.append((series[0]["xs"][0], f"CB min ~ {cb:.3f} eV", QtGui.QColor(160,255,160)))
        self.band_plot.set_markers(markers)
        if vb is not None and cb is not None and cb>vb:
            gap = cb - vb
            self.status.showMessage(f"Tahmini band aralığı ~ {gap:.3f} eV", 7000)
        else:
            self.status.showMessage("Band aralığı tespit edilemedi.", 5000)

    # -------------------------- Büyük görüntüleyici ------------------------- #
    def open_big_viewer(self):
        if not self.view.frames:
            QtWidgets.QMessageBox.information(self, "Bilgi", "Önce kare yükleyin."); return
        bg = color_from_choice(self.cb_bg.currentText())
        fg = text_from_choice(self.cb_fg.currentText())
        self.big = BigViewer(self.view.frames, self, bg=bg, fg=fg)
        # tarz aktar
        self.big.view.draw_bonds = self.view.draw_bonds
        self.big.view.draw_cell = self.view.draw_cell
        self.big.view.bond_scale = self.view.bond_scale
        self.big.view.bond_width = self.view.bond_width
        self.big.view.atom_radius_scale = self.view.atom_radius_scale
        self.big.view.downsample = self.view.downsample
        self.big.view.set_interval(self.sb_interval.value())
        self.big.show(); self.big.raise_()

    # ----------------------------- GIF/MP4 export --------------------------- #
    def export_animation(self, fmt: str = "gif"):
        if not self.view.frames:
            QtWidgets.QMessageBox.information(self, "Bilgi", "Önce kare yükleyin."); return
        fmt = fmt.lower()
        if fmt not in ("gif", "mp4"):
            QtWidgets.QMessageBox.warning(self, "Uyarı", "Format gif/mp4 olmalı."); return
        filt = "GIF (*.gif)" if fmt=="gif" else "MP4 (*.mp4)"
        out_path, _ = QtWidgets.QFileDialog.getSaveFileName(self, f"{fmt.upper()} kaydet", os.getcwd(), filt)
        if not out_path: return

        fps = self.sb_fps.value()
        a = max(1, min(self.sb_fr_a.value(), len(self.view.frames)))
        b = max(1, min(self.sb_fr_b.value(), len(self.view.frames)))
        if a > b: a, b = b, a
        scale = float(self.sp_export_scale.value())

        tmp_dir = tempfile.mkdtemp(prefix="qe_frames_")
        try:
            self.status.showMessage("Kareler oluşturuluyor…")
            k = 0; cur_idx = self.view.idx
            for i in range(a-1, b):
                self.view.idx = i
                img = self.view.render_to_qimage(scale_factor=scale)
                fn = os.path.join(tmp_dir, f"frame_{k:05d}.png")
                img.save(fn); k += 1
            self.view.idx = cur_idx

            ffmpeg = shutil.which("ffmpeg")
            if ffmpeg is None:
                QtWidgets.QMessageBox.information(self, "Bilgi",
                    f"ffmpeg bulunamadı. PNG kareler klasörde:\n{tmp_dir}\n"
                    f"Elinizle birleştirebilirsiniz.\nÖrnek:\n"
                    f"ffmpeg -y -framerate {fps} -i frame_%05d.png -vf palettegen palette.png\n"
                    f"ffmpeg -y -framerate {fps} -i frame_%05d.png -i palette.png -lavfi paletteuse out.gif")
                QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(tmp_dir)); return

            if fmt == "gif":
                palette = os.path.join(tmp_dir, "palette.png")
                cmd1 = [ffmpeg, "-y", "-framerate", str(fps), "-i", os.path.join(tmp_dir, "frame_%05d.png"),
                        "-vf", "palettegen", palette]
                cmd2 = [ffmpeg, "-y", "-framerate", str(fps), "-i", os.path.join(tmp_dir, "frame_%05d.png"),
                        "-i", palette, "-lavfi", "paletteuse", out_path]
                subprocess.run(cmd1, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                r = subprocess.run(cmd2, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            else:
                cmd = [ffmpeg, "-y", "-framerate", str(fps), "-i", os.path.join(tmp_dir, "frame_%05d.png"),
                       "-vcodec", "libx264", "-pix_fmt", "yuv420p", out_path]
                r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

            if os.path.isfile(out_path):
                self.status.showMessage(f"Dışa aktarıldı: {out_path}", 6000)
                QtWidgets.QMessageBox.information(self, "Bitti", f"Kaydedildi:\n{out_path}")
            else:
                QtWidgets.QMessageBox.warning(self, "Hata", "ffmpeg çıktı üretmedi. Kareler klasörde tutuldu.")
                QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(tmp_dir))
        finally:
            pass

    # ------------------------------ İç yardımcılar ------------------------- #
    def _adsorption_energy(self, system_out: Optional[str], surface_out: Optional[str], adsorbate_out: Optional[str]) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[float]]:
        Esys = self._parse_last_energy_ev(system_out) if system_out else None
        Esurf = self._parse_last_energy_ev(surface_out) if surface_out else None
        Eads = self._parse_last_energy_ev(adsorbate_out) if adsorbate_out else None
        if Esys is None or Esurf is None or Eads is None: return None, Esys, Esurf, Eads
        return Esys - Esurf - Eads, Esys, Esurf, Eads

    def _parse_last_energy_ev(self, path: str) -> Optional[float]:
        if not path or not os.path.isfile(path): return None
        re_e = re.compile(r'!\s*total energy\s*=\s*([\-0-9\.Ee\+]+)\s*Ry', re.I)
        last = None
        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            for ln in f:
                m = re_e.search(ln)
                if m: last = float(m.group(1)) * RY2EV
        return last

    def _parse_bader_acf_sum(self, path: str) -> Optional[float]:
        if not path or not os.path.isfile(path): return None
        try:
            with open(path, 'r') as f:
                lines = f.readlines()
            for ln in reversed(lines):
                if 'SUM' in ln and 'CHARGE' in ln.upper():
                    parts = ln.strip().split()
                    return float(parts[-1])
        except: return None
        return None

    def classify_site(self, frame: Frame) -> Tuple[Optional[str], Optional[str], Optional[float]]:
        if not frame.positions: return None, None, None
        zvals = [p[2] for p in frame.positions]
        zmax, zmin = max(zvals), min(zvals); span = (zmax - zmin) if zmax>zmin else 1.0
        thresh = zmax - 0.15*span
        surface = [i for i,p in enumerate(frame.positions) if p[2] >= thresh]
        if len(surface) < 1: return None, None, None
        min_d, min_i = 1e9, None
        for i in range(len(frame.positions)):
            di = min(math.dist(frame.positions[i], frame.positions[j]) for j in surface)
            if di < min_d: min_d, min_i = di, i
        if min_i is None: return None, None, None
        dlist = []
        for j in surface:
            d = math.dist(frame.positions[min_i], frame.positions[j]); dlist.append((d, j))
        dlist.sort(key=lambda x:x[0])
        nearest = dlist[:3]
        if len(nearest)==0: site = None
        elif len(nearest)==1 or (nearest[0][0] + 0.3 < nearest[1][0]): site = 'top'
        elif len(nearest)>=2 and abs(nearest[0][0]-nearest[1][0])<0.25 and (len(nearest)==2 or nearest[1][0] + 0.3 < nearest[2][0]): site = 'bridge'
        else: site = 'hollow'
        return site, frame.symbols[min_i], min_d

    def _parse_neb_images(self, paths: List[str]) -> Tuple[List[Frame], List[float]]:
        all_frames = []; energies = []
        for p in paths:
            if not os.path.isfile(p): continue
            fr, met, lastE = self.parser.parse_file(p)
            if fr: all_frames.append(fr[-1])
            energies.append(lastE if lastE is not None else float('nan'))
        return all_frames, energies

    # =============================== BET & İZOTERM ========================= #
    def bet_add_file(self):
        d = os.getcwd()
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "İzoterm dosyası ekle", d, "Tüm dosyalar (*.*)")
        if path:
            self.bet_multi.appendPlainText(path)

    def _load_isotherm_file(self, path: str) -> Tuple[List[float], List[float]]:
        xs, vs = [], []
        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            for ln in f:
                s = ln.strip()
                if not s or s.startswith('#'): continue
                parts = s.replace(',', ' ').replace(';',' ').split()
                if len(parts) < 2: continue
                try:
                    x = float(parts[0]); v = float(parts[1])
                except:
                    continue
                xs.append(x); vs.append(v)
        z = list(zip(xs,vs)); z.sort(key=lambda t:t[0])
        xs = [a for a,_ in z]; vs = [b for _,b in z]
        return xs, vs

    def bet_load_isotherm(self):
        path = self.le_bet.lineedit.text().strip()
        if not path or not os.path.isfile(path):
            QtWidgets.QMessageBox.information(self, "Bilgi", "Geçerli bir izoterm dosyası seçin."); return
        xs, vs = self._load_isotherm_file(path)
        if not xs:
            QtWidgets.QMessageBox.information(self, "Bilgi", "Dosyada veri bulunamadı (en az iki sütun: P/P0 ve V)."); return
        self._bet_raw = (xs, vs)
        ser = [{"xs":xs, "ys":vs if self.rb_bet_cm3.isChecked() else [v for v in vs], 
                "color":QtGui.QColor(0,200,255), "label":os.path.basename(path)}]
        self.bet_iso_plot.set_series_list(ser)
        xmin = float(self.sp_bet_xmin.value()); xmax=float(self.sp_bet_xmax.value())
        self.bet_iso_plot.set_markers([(xmin, "BET xmin", QtGui.QColor(255,140,0)), (xmax, "BET xmax", QtGui.QColor(255,140,0))])
        self.status.showMessage(f"İzoterm yüklendi: {len(xs)} nokta.", 5000)

    def _linreg(self, xs: List[float], ys: List[float]) -> Tuple[float,float,float]:
        n = len(xs)
        if n<2: return float('nan'), float('nan'), float('nan')
        sx = sum(xs); sy = sum(ys)
        sxx = sum(x*x for x in xs); sxy = sum(x*y for x,y in zip(xs,ys))
        denom = n*sxx - sx*sx
        if abs(denom) < 1e-15:
            return float('nan'), float('nan'), float('nan')
        m = (n*sxy - sx*sy) / denom
        b = (sy - m*sx) / n
        yhat = [m*x + b for x in xs]
        ymean = sy / n
        ss_res = sum((y - yh)**2 for y,yh in zip(ys,yhat))
        ss_tot = sum((y - ymean)**2 for y in ys)
        r2 = 1.0 - (ss_res/ss_tot if ss_tot>0 else float('inf'))
        return m, b, r2

    def _bet_transform(self, xs, vs_mol_per_g, xmin, xmax):
        x_sel, y_sel = [], []
        for x,v in zip(xs, vs_mol_per_g):
            if x<=0.0 or x>=1.0: continue
            if not (xmin <= x <= xmax): continue
            if v <= 0.0: continue
            y = 1.0 / (v * ((1.0/x) - 1.0))
            x_sel.append(x); y_sel.append(y)
        return x_sel, y_sel

    def bet_compute_single(self):
        raw = getattr(self, "_bet_raw", None)
        if not raw:
            QtWidgets.QMessageBox.information(self, "Bilgi", "Önce izoterm dosyasını yükleyin."); return
        xs, vs_in = raw
        # BET: mol/g’a çevir
        if self.rb_bet_cm3.isChecked():
            Vmol_stp = float(self.sp_bet_vmolar.value())
            vs_mol = [v / Vmol_stp for v in vs_in]  # mol/g
            vs_plot = vs_in[:]  # cm3/g
        else:
            vs_mol = vs_in[:]
            Vmol_stp = float(self.sp_bet_vmolar.value())
            vs_plot = [v*Vmol_stp for v in vs_in]
        xmin = float(self.sp_bet_xmin.value()); xmax = float(self.sp_bet_xmax.value())
        if xmax <= xmin: xmin, xmax = xmax, xmin
        x_sel, y_sel = self._bet_transform(xs, vs_mol, xmin, xmax)
        if len(x_sel) < 3:
            QtWidgets.QMessageBox.information(self, "Bilgi", "Seçilen aralıkta yeterli nokta yok (≥3)."); return
        m, b, r2 = self._linreg(x_sel, y_sel)
        Vm = 1.0 / (m + b)          # mol/g
        C  = 1.0 + m / b if abs(b) > 1e-15 else float('inf')
        sigma_nm2 = float(self.sp_bet_sigma.value())
        S = Vm * NA * (sigma_nm2 * 1e-18)
        Vm_cm3g = Vm * float(self.sp_bet_vmolar.value())
        T = float(self.sp_bet_T.value())
        dQ = (R_gas * T * math.log(C)) / 1000.0 if C>0 else float('nan')

        # grafikleri güncelle
        self.bet_iso_plot.set_series_list([{"xs":xs, "ys":vs_plot, "color":QtGui.QColor(0,200,255), "label":"V(P/P0)"}])
        self.bet_iso_plot.set_markers([(xmin, "BET xmin", QtGui.QColor(255,140,0)), (xmax, "BET xmax", QtGui.QColor(255,140,0))])
        xfit = [xmin, xmax]; yfit = [m*x + b for x in xfit]
        self.bet_plot.set_series_list([
            {"xs": x_sel, "ys": y_sel, "color": QtGui.QColor(120,220,120), "label":"BET data"},
            {"xs": xfit, "ys": yfit, "color": QtGui.QColor(255,200,30), "label":"fit"}])

        def fmt(v, p=6): 
            return f"{v:.{p}f}" if isinstance(v,(int,float)) and (v==v) and (abs(v) != float('inf')) else "N/A"
        html = []
        html.append("<b>BET Sonuçları (tek dosya)</b>")
        html.append(f"Aralık: P/P0 ∈ [{xmin:.3f}, {xmax:.3f}] (N={len(x_sel)}), R²={fmt(r2,4)}")
        html.append(f"Vm = <b>{fmt(Vm,6)}</b> mol/g  |  {fmt(Vm_cm3g,3)} cm³(STP)/g")
        html.append(f"C = <b>{fmt(C,4)}</b>  |  ΔQ ≈ RT ln C = <b>{fmt(dQ,3)} kJ/mol</b>")
        html.append(f"S_BET = <b>{fmt(S,3)} m²/g</b>")
        self.bet_summary.setHtml("<br>".join(html))
        self._bet_cache = {"xmin":xmin,"xmax":xmax,"m":m,"b":b,"r2":r2,"Vm":Vm,"Vm_cm3g":Vm_cm3g,"C":C,"S":S,"dQ_kJmol":dQ,"x_sel":x_sel,"y_sel":y_sel}

    # ---- Langmuir ve t-plot yardımcıları ---- #
    def _langmuir_fit(self, xs, vs_cm3g, xmin, xmax):
        xL, yL = [], []
        for x, v in zip(xs, vs_cm3g):
            if x<=0.0 or v<=0.0: continue
            if xmin<=x<=xmax:
                xL.append(x); yL.append(x / v)  # y = x/v
        if len(xL) < 3: 
            return None
        a, c, r2 = self._linreg(xL, yL)  # y = a*x + c
        if a==0 or not (a==a and c==c): return None
        v_m = 1.0 / a
        Kp = a/c if c!=0 else float('inf')  # K' = a/c
        xfit = [xmin, xmax]; yfit = [a*x + c for x in xfit]
        return {"x":xL, "y":yL, "a":a, "c":c, "r2":r2, "vm":v_m, "Kp":Kp, "xfit":xfit, "yfit":yfit}

    def _t_halsey(self, x):
        # t(nm) ≈ 0.354 * (5 / ln(P0/P))^(1/3)
        val = 0.354 * (5.0 / max(1e-12, math.log(1.0/x))) ** (1.0/3.0)
        return val

    def _t_harkins_jura(self, x):
        # t(Å) = sqrt(13.99 / (0.034 - log10(P/P0)))  → nm
        denom = 0.034 - (math.log10(x) if x>0 else -99.0)
        if denom <= 1e-9: denom = 1e-9
        tA = math.sqrt(13.99 / denom)
        return 0.1*tA  # nm

    def _tplot_fit(self, xs, vs_cm3g, xmin, xmax, model="Harkins–Jura"):
        xt, vt = [], []
        for x, v in zip(xs, vs_cm3g):
            if xmin<=x<=xmax and 0.0<x<1.0:
                t = self._t_harkins_jura(x) if "Harkins" in model else self._t_halsey(x)
                xt.append(t); vt.append(v)
        if len(xt) < 3: return None
        m, b, r2 = self._linreg(xt, vt)  # V = m t + b
        return {"t":xt, "V":vt, "m":m, "b":b, "r2":r2}

    def bet_run_multi(self):
        lines = [ln.strip() for ln in self.bet_multi.toPlainText().splitlines() if ln.strip()]
        if not lines:
            QtWidgets.QMessageBox.information(self, "Bilgi", "Çoklu analiz için en az bir dosya girin."); return

        Vmol_stp = float(self.sp_bet_vmolar.value())
        bet_xmin, bet_xmax = float(self.sp_bet_xmin.value()), float(self.sp_bet_xmax.value())
        L_xmin, L_xmax = float(self.sp_L_xmin.value()), float(self.sp_L_xmax.value())
        t_xmin, t_xmax = float(self.sp_t_xmin.value()), float(self.sp_t_xmax.value())
        t_model = self.cb_t_model.currentText()

        iso_series = []; bet_series = []; lang_series = []; tplot_series = []
        summary_lines = []
        color_idx = 0

        for line in lines:
            path = line
            label = None
            if "," in line:
                sp = [s.strip() for s in line.split(",",1)]
                path = sp[0]; label = sp[1] or None
            if not os.path.isfile(path): 
                summary_lines.append(f"<i>Atlandı (bulunamadı)</i>: {line}")
                continue

            xs, vs_in = self._load_isotherm_file(path)
            if not xs: 
                summary_lines.append(f"<i>Boş/uygunsuz</i>: {line}")
                continue

            # grafikte cm3/g göstereceğiz
            if self.rb_bet_cm3.isChecked():
                vs_cm3g = vs_in[:]
                vs_mol = [v / Vmol_stp for v in vs_in]
            else:
                vs_mol = vs_in[:]
                vs_cm3g = [v * Vmol_stp for v in vs_in]

            tag = label or os.path.basename(path)
            col = self.colors[color_idx % len(self.colors)]; color_idx += 1

            # izoterm
            iso_series.append({"xs":xs, "ys":vs_cm3g, "color":col, "label":tag})

            # BET
            x_sel, y_sel = self._bet_transform(xs, vs_mol, bet_xmin, bet_xmax)
            if len(x_sel) >= 3:
                m, b, r2 = self._linreg(x_sel, y_sel)
                Vm = 1.0 / (m + b)
                C  = 1.0 + m / b if abs(b) > 1e-15 else float('inf')
                S  = Vm * NA * (float(self.sp_bet_sigma.value()) * 1e-18)
                Vm_cm3g = Vm * Vmol_stp
                T = float(self.sp_bet_T.value())
                dQ = (R_gas*T*math.log(C))/1000.0 if C>0 else float('nan')
                xfit = [bet_xmin, bet_xmax]; yfit = [m*x + b for x in xfit]
                bet_series.append({"xs":x_sel, "ys":y_sel, "color":col, "label":f"{tag} (BET)"})
                bet_series.append({"xs":xfit, "ys":yfit, "color":QtGui.QColor(255,200,30), "label":f"{tag} fit"})
                summary_lines.append(f"<b>{tag}</b> — BET: Vm={Vm:.6f} mol/g ({Vm_cm3g:.3f} cm³/g), C={C:.2f}, S={S:.2f} m²/g, R²={r2:.4f}, ΔQ≈{dQ:.2f} kJ/mol")
            else:
                summary_lines.append(f"<b>{tag}</b> — BET: <i>yetersiz nokta</i>")

            # Langmuir
            L = self._langmuir_fit(xs, vs_cm3g, L_xmin, L_xmax)
            if L:
                lang_series.append({"xs":L["x"], "ys":L["y"], "color":col, "label":f"{tag} (x/V)"})
                lang_series.append({"xs":L["xfit"], "ys":L["yfit"], "color":QtGui.QColor(255,200,30), "label":f"{tag} fit"})
                summary_lines.append(f"&nbsp;&nbsp;Langmuir: v_m={L['vm']:.3f} cm³/g, K'={L['Kp']:.3f}, R²={L['r2']:.4f}")
            else:
                summary_lines.append(f"&nbsp;&nbsp;Langmuir: <i>yetersiz/uygunsuz</i>")

            # t-plot
            TPL = self._tplot_fit(xs, vs_cm3g, t_xmin, t_xmax, model=t_model)
            if TPL:
                tfit_x = [min(TPL["t"]), max(TPL["t"])]; tfit_y = [TPL["m"]*t + TPL["b"] for t in tfit_x]
                tplot_series.append({"xs":TPL["t"], "ys":TPL["V"], "color":col, "label":f"{tag} (V-t)"})
                tplot_series.append({"xs":tfit_x, "ys":tfit_y, "color":QtGui.QColor(255,200,30), "label":f"{tag} fit"})
                summary_lines.append(f"&nbsp;&nbsp;t-plot({t_model}): V= m t + b ⇒ b≈{TPL['b']:.3f} cm³/g (V_micro), m≈{TPL['m']:.3f} (cm³/g)/nm, R²={TPL['r2']:.4f}")
            else:
                summary_lines.append(f"&nbsp;&nbsp;t-plot: <i>yetersiz/uygunsuz</i>")

        # grafikleri bas
        self.bet_iso_plot.set_series_list(iso_series)
        self.bet_iso_plot.set_markers([(bet_xmin, "BET xmin", QtGui.QColor(255,140,0)), (bet_xmax, "BET xmax", QtGui.QColor(255,140,0))])
        self.bet_plot.set_series_list(bet_series)
        self.bet_lang_plot.set_series_list(lang_series)
        self.bet_lang_plot.set_markers([(L_xmin, "xmin", QtGui.QColor(255,140,0)), (L_xmax, "xmax", QtGui.QColor(255,140,0))])
        self.bet_t_plot.set_series_list(tplot_series)
        self.bet_t_plot.set_markers([])

        self.bet_summary.setHtml("<br>".join(summary_lines))
        self.status.showMessage("Çoklu izoterm analizi tamamlandı.", 6000)

    def bet_save_csv(self):
        cache = getattr(self, "_bet_cache", None)
        raw = getattr(self, "_bet_raw", None)
        if not cache or not raw:
            QtWidgets.QMessageBox.information(self, "Bilgi", "Önce BET (tek dosya) hesaplayın."); return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "BET CSV kaydet", os.path.join(os.getcwd(),"bet_results.csv"), "CSV (*.csv)")
        if not path: return
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["# BET sonuçları"])
            for k in ["xmin","xmax","m","b","r2","Vm","Vm_cm3g","C","S","dQ_kJmol"]:
                w.writerow([k, cache[k]])
            w.writerow(["# BET verisi (x_sel, y_sel)"])
            w.writerow(["x", "y"])
            for x,y in zip(cache["x_sel"], cache["y_sel"]):
                w.writerow([x, y])
        self.status.showMessage(f"BET CSV kaydedildi: {path}", 5000)

# --------------------------------- main ------------------------------------ #
def main():
    app = QtWidgets.QApplication(sys.argv)
    app.setApplicationName("QE Suite (Qt5)")
    app.setStyle("Fusion")

    w = MainWindow()

    # Varsayılan: koyu tema
    bg = color_from_choice("Siyah"); fg = text_from_choice("Beyaz")
    w.apply_theme(bg, fg)

    w.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
