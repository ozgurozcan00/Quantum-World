# -*- coding: utf-8 -*-
"""
FTIR Designer — PyQt5 (Ultra Pro)

Neler yeni?
- Çok daha stabil: *yıldız import* kaldırıldı, giriş kontrolü ve hata yakalama eklendi
- UI donmalarını azaltan **debounce** (50 ms) yeniden-hesaplama zamanlayıcısı
- Aşırı örnekleme/Nt*M büyükse performans uyarısı ve güvenli sınırlar
- CSV okuma/çıktı için sağlamlaştırılmış kod (kodlama algılama, başlık satırı toleransı)
- Basit **preset** kaydet/yükle (JSON) — aynı tasarımı tekrar çağırın
- 2B/3B çizim ve FFT tarafında sayısal kararlılık iyileştirmeleri
- Kod yapısı sadeleştirildi, tekrar eden yardımcılar kaldırıldı

Bağımlılıklar: PyQt5, numpy, matplotlib
"""
from __future__ import annotations

import csv
import json
import math
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import QTimer, Qt

# Matplotlib → PyQt5 tuvali
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

# =========================== MODEL & YARDIMCILAR ===========================

MAX_POINTS = 1_200_000  # Nt + zero-fill için güvenli üst sınır (yaklaşık)
RECOMPUTE_DEBOUNCE_MS = 50


def gas_bands(name: str) -> List[Dict[str, float]]:
    n = name.strip().lower()
    if n in ("ethanol", "c2h5oh"):
        return [dict(c=1050, fwhm=50, amp=0.25), dict(c=2950, fwhm=150, amp=0.15)]
    if n == "co":
        return [dict(c=2143, fwhm=6, amp=0.25)]
    if n in ("co2", "co₂"):
        return [dict(c=2349, fwhm=20, amp=0.35), dict(c=667, fwhm=20, amp=0.35)]
    if n == "ch4":
        return [dict(c=3019, fwhm=40, amp=0.20), dict(c=1306, fwhm=30, amp=0.15)]
    if n == "hcn":
        return [dict(c=3312, fwhm=20, amp=0.25), dict(c=2089, fwhm=12, amp=0.15)]
    if n == "dmmp":
        return [dict(c=1260, fwhm=50, amp=0.30), dict(c=1040, fwhm=40, amp=0.18)]
    if n in ("2-cees", "cees", "chloroethylsulfide"):
        return [dict(c=1250, fwhm=60, amp=0.25), dict(c=840, fwhm=40, amp=0.20)]
    return []


def gaussian(x: np.ndarray, x0: float, fwhm: float, A: float) -> np.ndarray:
    s = fwhm / (2.0 * math.sqrt(2.0 * math.log(2.0)))
    return A * np.exp(-((x - x0) ** 2) / (2.0 * s * s))


def gas_absorbance(sigma: np.ndarray, name: str) -> np.ndarray:
    A = np.zeros_like(sigma, dtype=float)
    for b in gas_bands(name):
        A += gaussian(sigma, b["c"], b["fwhm"], b["amp"])
    return A


def build_mixture_absorbance(
    sigma: np.ndarray, scales: Dict[str, float], L_m: float, passes: int
) -> np.ndarray:
    A = np.zeros_like(sigma, dtype=float)
    for g, k in scales.items():
        if k == 0:
            continue
        A += float(k) * gas_absorbance(sigma, g)
    Leff = max(0.0, L_m) * max(1, int(round(passes)))
    return A * Leff


def window_array(N: int, kind: str, alpha: float) -> np.ndarray:
    if kind == "boxcar":
        return np.ones(N)
    if kind == "hamming":
        n = np.arange(N)
        return 0.54 - 0.46 * np.cos(2 * np.pi * n / (N - 1))
    if kind == "blackman":
        n = np.arange(N)
        return 0.42 - 0.5 * np.cos(2 * np.pi * n / (N - 1)) + 0.08 * np.cos(
            4 * np.pi * n / (N - 1)
        )
    if kind == "gaussian":
        n = np.arange(N)
        x = (n - (N - 1) / 2) / ((N - 1) / 2)
        return np.exp(-alpha * x * x)
    return np.ones(N)


@dataclass
class VoltsOptions:
    baseV: float = 1.2
    burstV: float = 0.8
    farV: float = 0.05
    noise_mV: float = 3.0
    fc: float = 5000.0  # Hz, 1st-order LPF
    drift_on: bool = False
    drift_mV: float = 1.0


def volts_map(t: np.ndarray, i: np.ndarray, opt: VoltsOptions) -> np.ndarray:
    base = opt.baseV
    bScale = opt.burstV
    fScale = opt.farV
    noise = opt.noise_mV / 1e3
    zpd = np.exp(-((t / 0.010) ** 2))
    far = 1.0 - zpd
    v = base + (bScale * zpd + fScale * far) * i
    if opt.drift_on and opt.drift_mV > 0:
        rng = np.random.default_rng(42)
        rnd = rng.normal(0.0, opt.drift_mV / 1e3 / 200.0, size=v.shape)
        v = np.cumsum(rnd) + v - np.mean(v) + base
    if opt.fc > 0:
        dt = t[1] - t[0]
        alpha = dt / (1.0 / (2 * np.pi * opt.fc) + dt)
        acc = v[0]
        out = np.empty_like(v)
        for k in range(v.size):
            acc = acc + alpha * (v[k] - acc)
            out[k] = acc
        v = out
    if noise > 0:
        rng = np.random.default_rng(123)
        v = v + rng.uniform(-noise, noise, size=v.shape)
    return v


@dataclass
class SynthResult:
    t: np.ndarray
    I0: np.ndarray
    I: np.ndarray
    V0: np.ndarray
    V: np.ndarray


def synthesize_interferogram(
    sigma: np.ndarray,
    A_abs: np.ndarray,
    res_cmin1: float,
    v_cm_s: float,
    Nt: int,
    volts_opt: VoltsOptions,
    x_static_cm: float = 0.0,  # hareketli aynanın statik konumu (cm)
) -> SynthResult:
    # kaynak spektrumu S(σ): hafif eğimli continuum
    S = 0.6 + 0.4 * (sigma - sigma[0]) / (sigma[-1] - sigma[0] + 1e-15)
    T = 10.0 ** (-A_abs)  # Transmittans

    OPDmax = 1.0 / max(res_cmin1, 1e-9)
    tspan = OPDmax / (2.0 * max(v_cm_s, 1e-9))
    t = np.linspace(-tspan, +tspan, max(Nt, 4))

    # Statik ayna konumu x_static (cm) eklenince OPD(t) = 2*(v*t + x_static)
    # faz terimindeki zaman ofseti: t' = t + x_static / v
    t_eff = t + (x_static_cm / max(1e-12, v_cm_s))

    omega = 2 * np.pi * (2.0 * v_cm_s) * sigma  # 2*v*σ

    # Outer-product yaklaşımı — büyük boyutlarda dikkat!
    I0 = np.cos(np.outer(t_eff, omega)).dot(S)
    I = np.cos(np.outer(t_eff, omega)).dot(S * T)

    I0 /= np.max(np.abs(I0)) + 1e-12
    I /= np.max(np.abs(I0)) + 1e-12

    V0 = volts_map(t, I0, volts_opt)
    V = volts_map(t, I, volts_opt)
    return SynthResult(t=t, I0=I0, I=I, V0=V0, V=V)


def spectrum_from_traces(
    t: np.ndarray,
    V0: np.ndarray,
    V: np.ndarray,
    v_cm_s: float,
    win_kind: str,
    galpha: float,
    zf: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    N = t.size
    w = window_array(N, win_kind, galpha)
    V0p = (V0 - np.mean(V0)) * w
    Vp = (V - np.mean(V)) * w

    Z = max(1, int(zf))
    Nf = 1 << int(math.ceil(math.log2(max(4, N * Z))))

    # güvenli bellek: çok büyükse kırp
    if Nf > MAX_POINTS:
        Nf = 1 << int(math.floor(math.log2(MAX_POINTS)))

    V0z = np.zeros(Nf)
    Vz = np.zeros(Nf)
    V0z[: min(N, Nf)] = V0p[: min(N, Nf)]
    Vz[: min(N, Nf)] = Vp[: min(N, Nf)]

    F0 = np.fft.rfft(V0z)
    F1 = np.fft.rfft(Vz)
    dt = t[1] - t[0]
    freqs = np.fft.rfftfreq(Nf, d=max(dt, 1e-12))
    sigma = freqs / (2.0 * max(v_cm_s, 1e-12))
    eps = 1e-15
    Tmag = (np.abs(F1) + eps) / (np.abs(F0) + eps)
    Amag = -np.log10(np.maximum(Tmag, eps))
    return sigma, Tmag, Amag


def nyquist_advisor(
    lambdaMin_um: float, v_cm_s: float, Nt: int, res: float
) -> Dict[str, float]:
    sigmaMax = 1.0 / (lambdaMin_um * 1e-4 + 1e-20)
    OPDmax = 1.0 / max(res, 1e-12)
    tspan = OPDmax / (2.0 * max(v_cm_s, 1e-9))
    dt = (2 * tspan) / max((Nt - 1), 1)
    fmax = 2.0 * v_cm_s * sigmaMax
    fs = 1.0 / max(dt, 1e-12)
    dOPD = 2.0 * v_cm_s * dt
    ny_ok = dOPD <= 1.0 / (2.0 * sigmaMax + 1e-20)
    return dict(fmax=fmax, fs=fs, fsSuggest=5 * fmax, dOPD=dOPD, nyOK=ny_ok)


def choose_target_bands(
    gas: str, sigma_min: float, sigma_max: float, mode: str = "auto"
) -> List[Dict[str, float]]:
    bands = [b for b in gas_bands(gas) if sigma_min <= b["c"] <= sigma_max]
    if not bands:
        return []
    if mode == "all":
        return bands
    bands.sort(key=lambda b: b["amp"], reverse=True)
    return [bands[0]]


# =========================== NNLS (SciPy gerektirmez) ===========================


def nnls_pg(A: np.ndarray, y: np.ndarray, max_iter: int = 3000, tol: float = 1e-6) -> np.ndarray:
    m, n = A.shape
    c = np.zeros(n)
    try:
        L = np.linalg.norm(A, 2) ** 2
    except Exception:
        L = np.linalg.norm(A) ** 2 + 1e-9
    step = 1.0 / max(L, 1e-9)
    AT = A.T
    for _ in range(max_iter):
        r = A.dot(c) - y
        g = AT.dot(r)
        c_new = c - step * g
        c_new[c_new < 0] = 0.0
        if np.linalg.norm(c_new - c) < tol * (np.linalg.norm(c) + 1e-12):
            c = c_new
            break
        c = c_new
    return c


# =========================== PyQt Tuvali ===========================


class MplCanvas(FigureCanvas):
    def __init__(self, title: str = "", parent=None, width=5, height=3, dpi=100, is3d=False):
        fig = Figure(figsize=(width, height), dpi=dpi)
        if is3d:
            self.ax = fig.add_subplot(111, projection="3d")
        else:
            self.ax = fig.add_subplot(111)
        self.ax.set_title(title)
        super().__init__(fig)
        self.setParent(parent)


# =========================== 2B Optik Şema ===========================


class LayoutView(QtWidgets.QGraphicsView):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setRenderHints(
            QtGui.QPainter.Antialiasing | QtGui.QPainter.SmoothPixmapTransform
        )
        self.scene = QtWidgets.QGraphicsScene(self)
        self.setScene(self.scene)
        self.w = 900
        self.h = 520
        self.setSceneRect(0, 0, self.w, self.h)

    def draw_layout(
        self,
        beam_d_mm: float,
        bs_d_mm: float,
        bs_angle_deg: float,
        oap_f_mm: float,
        warn_fit: bool,
    ) -> None:
        self.scene.clear()
        margin = 60
        y0 = self.h / 2
        px_per_mm = 4.0
        beam_w = max(1.5, beam_d_mm * px_per_mm * 0.15)
        beam_pen = QtGui.QPen(QtGui.QColor("#1f77b4"), beam_w, Qt.SolidLine, Qt.RoundCap)
        if warn_fit:
            beam_pen = QtGui.QPen(QtGui.QColor("#b00020"), beam_w, Qt.SolidLine, Qt.RoundCap)

        src_x = margin
        self.scene.addRect(
            src_x - 10,
            y0 - 15,
            20,
            30,
            QtGui.QPen(Qt.black),
            QtGui.QBrush(QtGui.QColor("#ffeacc")),
        )
        self.scene.addText("Kaynak").setPos(src_x - 20, y0 + 20)

        oap_x = src_x + 100
        oap_r = 30
        self.scene.addEllipse(
            oap_x - oap_r,
            y0 - oap_r,
            2 * oap_r,
            2 * oap_r,
            QtGui.QPen(QtGui.QColor("#444")),
            QtGui.QBrush(QtGui.QColor("#e6f2ff")),
        )
        self.scene.addText(f"OAP f={oap_f_mm:.0f} mm").setPos(oap_x - 50, y0 - 60)

        bs_x = oap_x + 180
        bs_h = bs_d_mm * px_per_mm
        bs_w = bs_h * 0.15 + 14
        bs_rect = QtCore.QRectF(bs_x - bs_w / 2, y0 - bs_h / 2, bs_w, bs_h)
        bs_item = self.scene.addRect(
            bs_rect,
            QtGui.QPen(QtGui.QColor("#333")),
            QtGui.QBrush(QtGui.QColor("#ddd")),
        )
        rot = QtGui.QTransform()
        rot.translate(bs_x, y0)
        rot.rotate(-bs_angle_deg)
        rot.translate(-bs_x, -y0)
        bs_item.setTransform(rot)
        self.scene.addText(f"BS Ø={bs_d_mm:.1f} mm @ {bs_angle_deg:.0f}°").setPos(
            bs_x - 80, y0 + 40
        )

        arm = 200
        m1_x = bs_x + arm
        m1_y = y0
        m2_x = bs_x
        m2_y = y0 - arm
        self.scene.addRect(
            m1_x - 8,
            m1_y - 35,
            16,
            70,
            QtGui.QPen(QtGui.QColor("#666")),
            QtGui.QBrush(QtGui.QColor("#f0f0f0")),
        )
        self.scene.addText("Sabit ayna").setPos(m1_x - 28, m1_y + 40)
        self.scene.addRect(
            m2_x - 35,
            m2_y - 8,
            70,
            16,
            QtGui.QPen(QtGui.QColor("#666")),
            QtGui.QBrush(QtGui.QColor("#f0f0f0")),
        )
        self.scene.addText("Hareketli ayna").setPos(m2_x + 40, m2_y - 18)

        det_x = bs_x - arm - 80
        self.scene.addRect(
            det_x - 12,
            y0 - 18,
            24,
            36,
            QtGui.QPen(Qt.black),
            QtGui.QBrush(QtGui.QColor("#e7ffe7")),
        )
        self.scene.addText("Dedektör").setPos(det_x - 26, y0 + 22)

        # ışın yolları
        self.scene.addLine(oap_x + 30, y0, bs_x - 5, y0, beam_pen)
        self.scene.addLine(bs_x + 5, y0, m1_x - 10, y0, beam_pen)
        self.scene.addLine(m1_x - 10, y0, bs_x + 5, y0, beam_pen)
        self.scene.addLine(bs_x - 5, y0, det_x + 12, y0, beam_pen)
        self.scene.addLine(bs_x, y0 - 5, bs_x, m2_y + 8, beam_pen)
        self.scene.addLine(bs_x, m2_y + 8, bs_x, y0 - 5, beam_pen)
        self.scene.addText(f"Beam Ø ~ {beam_d_mm:.1f} mm").setPos(oap_x + 40, y0 - 70)


# =========================== Multipass Hücre (Herriott/White) ===========================


class CellCanvas(QtWidgets.QGraphicsView):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setRenderHints(
            QtGui.QPainter.Antialiasing | QtGui.QPainter.SmoothPixmapTransform
        )
        self.scene = QtWidgets.QGraphicsScene(self)
        self.setScene(self.scene)
        self.setSceneRect(0, 0, 900, 520)

    def draw_herriott(
        self,
        R_mm,
        L_mm,
        D_mm,
        hole_r_mm,
        alpha_mrad,
        beta_mrad,
        x0_mm,
        y0_mm,
        max_bounce,
        spotsA,
        spotsB,
    ) -> None:
        self.scene.clear()
        W, H = 900, 520
        self.setSceneRect(0, 0, W, H)
        cxA, cyA = 250, H / 2
        cxB, cyB = cxA + 1.2 * L_mm, H / 2
        scale = 2.0
        rad = (D_mm / 2.0) * scale
        self.scene.addEllipse(
            cxA - rad,
            cyA - rad,
            2 * rad,
            2 * rad,
            QtGui.QPen(Qt.black, 1.2),
            QtGui.QBrush(QtGui.QColor("#fafafa")),
        )
        self.scene.addText("Ayna A").setPos(cxA - 28, cyA + rad + 6)
        self.scene.addEllipse(
            cxB - rad,
            cyB - rad,
            2 * rad,
            2 * rad,
            QtGui.QPen(Qt.black, 1.2),
            QtGui.QBrush(QtGui.QColor("#fafafa")),
        )
        self.scene.addText("Ayna B").setPos(cxB - 28, cyB + rad + 6)
        self.scene.addEllipse(
            cxA + (x0_mm) * scale - 3,
            cyA - (y0_mm) * scale - 3,
            6,
            6,
            QtGui.QPen(QtGui.QColor("#d2691e")),
            QtGui.QBrush(QtGui.QColor("#d2691e")),
        )
        self.scene.addText("Giriş/Çıkış").setPos(
            cxA + (x0_mm) * scale + 6, cyA - (y0_mm) * scale - 8
        )

        def draw(spots, cx, cy, color):
            for i, (x, y) in enumerate(spots):
                px = cx + x * scale
                py = cy - y * scale
                self.scene.addEllipse(
                    px - 3,
                    py - 3,
                    6,
                    6,
                    QtGui.QPen(QtGui.QColor(color)),
                    QtGui.QBrush(QtGui.QColor(color)),
                )
                if i == 0:
                    self.scene.addText("1").setPos(px + 3, py - 10)

        draw(spotsA, cxA, cyA, "#1f77b4")
        draw(spotsB, cxB, cyB, "#d62728")
        txt = (
            f"Herriott: R={R_mm:.0f} mm, L={L_mm:.0f} mm, D={D_mm:.0f} mm, delik r={hole_r_mm:.1f} mm\n"
            f"α={alpha_mrad:.1f} mrad, β={beta_mrad:.1f} mrad, max={max_bounce}"
        )
        self.scene.addText(txt).setPos(20, 10)

    def draw_white(
        self, R_mm, L_mm, D_mm, alpha_mrad, beta_mrad, max_bounce, spots
    ) -> None:
        self.scene.clear()
        W, H = 900, 520
        self.setSceneRect(0, 0, W, H)
        scale = 2.0
        cx = [250, 250 + L_mm * scale * 0.9, 250 + L_mm * scale * 0.45]
        cy = [H / 2 - 140, H / 2 - 140, H / 2 + 120]
        rad = (D_mm / 2.0) * scale
        for i in range(3):
            self.scene.addEllipse(
                cx[i] - rad,
                cy[i] - rad,
                2 * rad,
                2 * rad,
                QtGui.QPen(Qt.black, 1.2),
                QtGui.QBrush(QtGui.QColor("#fafafa")),
            )
            self.scene.addText(f"Ayna {i+1}").setPos(cx[i] - 28, cy[i] + rad + 6)
        for k, (mx, my) in enumerate(spots):
            px = mx * scale
            py = my * scale
            self.scene.addEllipse(
                px - 3,
                py - 3,
                6,
                6,
                QtGui.QPen(QtGui.QColor("#2ca02c")),
                QtGui.QBrush(QtGui.QColor("#2ca02c")),
            )
            if k == 0:
                self.scene.addText("1").setPos(px + 4, py - 10)
        self.scene.addText(
            f"White (yaklaşık): R={R_mm:.0f} mm, L≈{L_mm:.0f} mm, D={D_mm:.0f} mm, α={alpha_mrad:.1f} mrad, β={beta_mrad:.1f} mrad"
        ).setPos(20, 10)


def herriott_trace(
    R_mm,
    L_mm,
    D_mm,
    hole_r_mm,
    alpha_mrad,
    beta_mrad,
    x0_mm,
    y0_mm,
    max_bounce=200,
):
    R = R_mm
    L = L_mm
    rad = D_mm / 2.0
    x = x0_mm
    y = y0_mm
    thx = alpha_mrad / 1000.0
    thy = beta_mrad / 1000.0
    spotsA = [(x, y)]
    spotsB: List[Tuple[float, float]] = []
    bounces = 0
    onA = True
    for _ in range(max_bounce):
        if onA:
            xB = x + L * thx
            yB = y + L * thy
            if xB**2 + yB**2 > rad**2:
                break
            spotsB.append((xB, yB))
            thx = thx - 2.0 * xB / R
            thy = thy - 2.0 * yB / R
            x, y = xB, yB
            onA = False
        else:
            xA = x + L * thx
            yA = y + L * thy
            if xA**2 + yA**2 > rad**2:
                break
            if ((xA - x0_mm) ** 2 + (yA - y0_mm) ** 2) <= (hole_r_mm**2) and bounces > 2:
                spotsA.append((xA, yA))
                bounces += 1
                break
            spotsA.append((xA, yA))
            thx = thx - 2.0 * xA / R
            thy = thy - 2.0 * yA / R
            x, y = xA, yA
            onA = True
        bounces += 1
    return spotsA, spotsB, bounces


def white_trace_simple(R_mm, L_mm, D_mm, alpha_mrad, beta_mrad, max_bounce=60):
    c1 = np.array([250.0, 120.0])
    c2 = np.array([250.0 + 0.9 * L_mm, 120.0])
    c3 = np.array([250.0 + 0.45 * L_mm, 120.0 + 0.85 * L_mm])
    centers = [c1, c2, c3]
    rad = D_mm / 2.0
    pos = c1 + np.array([10.0, 0.0])
    dirv = np.array([alpha_mrad / 5.0, beta_mrad / 5.0])
    spots: List[Tuple[float, float]] = []
    mirror_idx = 0
    for _ in range(max_bounce):
        nxt = (mirror_idx + 1) % 3
        tgt = centers[nxt]
        d = tgt - pos
        if np.linalg.norm(d) < 1e-6:
            break
        step = d / np.linalg.norm(d)
        pos = tgt - step * (rad - 2.0)
        spots.append((pos[0], pos[1]))
        n = (pos - centers[nxt])
        n = n / (np.linalg.norm(n) + 1e-12)
        dirv = dirv - 2.0 * np.dot(dirv, n) * n
        pos = pos + dirv * 10.0
        mirror_idx = nxt
    return spots


# =========================== SNR & LOD ===========================


def snr_lod_calc(Vsig, NEP_WrtHz, Rv_VperW, BW_Hz=None, Tint_s=None):
    if Tint_s is not None and Tint_s > 0:
        BW = 1.0 / (2.0 * Tint_s)
    else:
        BW = max(1.0, BW_Hz if BW_Hz is not None else 1000.0)
    Vn = Rv_VperW * NEP_WrtHz * math.sqrt(BW)
    snr = Vsig / max(Vn, 1e-15)
    return snr, Vn


def lod_from_beer(A_line, Leff_m, S_sens_per_conc, snr_required, snr_actual, conc_ref=1.0):
    Leff_cm = Leff_m * 100.0
    denom = max(1e-12, S_sens_per_conc * Leff_cm)
    lod = (snr_required / max(1e-12, snr_actual)) * (A_line / denom) * conc_ref
    return max(0.0, lod)


# =========================== 3B Optik Yardımcıları ===========================


def rot_z(angle_deg: float) -> np.ndarray:
    th = np.deg2rad(angle_deg)
    c = np.cos(th)
    s = np.sin(th)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=float)


def reflect(v: np.ndarray, n: np.ndarray) -> np.ndarray:
    n = n / (np.linalg.norm(n) + 1e-12)
    return v - 2.0 * np.dot(v, n) * n


def line_plane_intersect(
    p0: np.ndarray, v: np.ndarray, p_plane: np.ndarray, n: np.ndarray
) -> Optional[np.ndarray]:
    denom = np.dot(v, n)
    if abs(denom) < 1e-9:
        return None
    t = np.dot(p_plane - p0, n) / denom
    if t <= 1e-9:
        return None
    return p0 + t * v


def segment(ax3d, p: np.ndarray, q: np.ndarray, **kw) -> None:
    ax3d.plot([p[0], q[0]], [p[1], q[1]], [p[2], q[2]], **kw)


# =========================== ANA PENCERE ===========================


class FTIRWindow(QtWidgets.QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("FTIR Designer — PyQt5 (Ultra Pro)")
        self.resize(1650, 980)

        # Menü: Preset kaydet/yükle
        self._build_menu()

        # ------ Sol: Kontroller ------
        self.ctrl = QtWidgets.QWidget()
        self.ctrl_layout = QtWidgets.QVBoxLayout(self.ctrl)
        self.ctrl_layout.setAlignment(Qt.AlignTop)

        # 1) Alım & Optikler
        box1 = QtWidgets.QGroupBox("1) Alım & Optikler")
        g1 = QtWidgets.QGridLayout(box1)
        self.sp_lamMin = self._spin_double(2.0, 1.0, 60.0, 0.1)
        self.sp_lamMax = self._spin_double(14.0, 1.0, 80.0, 0.1)
        self.sp_res = self._spin_double(1.0, 0.05, 10.0, 0.05)
        self.sp_v = self._spin_double(1.0, 0.01, 20.0, 0.05)
        self.sp_fcol = self._spin_double(150, 10, 2000, 1)
        self.sp_bsd = self._spin_double(12.7, 1, 100, 0.1)
        self.sp_bsang = self._spin_double(45, 0, 90, 1)
        self.sp_bsmargin = self._spin_double(0.5, 0, 5, 0.1)
        self.cb_override = QtWidgets.QCheckBox("Beam Ø override")
        self.sp_beamDia = self._spin_double(9.0, 0.5, 50, 0.5)
        g1.addWidget(QtWidgets.QLabel("Bant min (µm)"), 0, 0)
        g1.addWidget(self.sp_lamMin, 0, 1)
        g1.addWidget(QtWidgets.QLabel("Bant max (µm)"), 0, 2)
        g1.addWidget(self.sp_lamMax, 0, 3)
        g1.addWidget(QtWidgets.QLabel("Δσ (cm⁻¹)"), 1, 0)
        g1.addWidget(self.sp_res, 1, 1)
        g1.addWidget(QtWidgets.QLabel("Ayna hızı v (cm/s)"), 1, 2)
        g1.addWidget(self.sp_v, 1, 3)
        g1.addWidget(QtWidgets.QLabel("OAP f_col (mm)"), 2, 0)
        g1.addWidget(self.sp_fcol, 2, 1)
        g1.addWidget(QtWidgets.QLabel("BS açıklık Ø (mm)"), 2, 2)
        g1.addWidget(self.sp_bsd, 2, 3)
        g1.addWidget(QtWidgets.QLabel("BS açısı (°)"), 3, 0)
        g1.addWidget(self.sp_bsang, 3, 1)
        g1.addWidget(QtWidgets.QLabel("BS marj (mm)"), 3, 2)
        g1.addWidget(self.sp_bsmargin, 3, 3)
        g1.addWidget(self.cb_override, 4, 0, 1, 2)
        g1.addWidget(QtWidgets.QLabel("Seçilen Beam Ø (mm)"), 4, 2)
        g1.addWidget(self.sp_beamDia, 4, 3)

        # 2) Hücre & Dedektör
        box2 = QtWidgets.QGroupBox("2) Gaz Hücresi & Dedektör")
        g2 = QtWidgets.QGridLayout(box2)
        self.sp_Lcell = self._spin_double(0.5, 0.0, 50.0, 0.05)
        self.sp_passes = self._spin_int(10, 1, 1000, 1)
        self.sp_baseV = self._spin_double(1.2, 0.0, 10.0, 0.1)
        self.sp_burstV = self._spin_double(0.8, 0.0, 10.0, 0.1)
        self.sp_farV = self._spin_double(0.05, 0.0, 1.0, 0.01)
        self.sp_noise = self._spin_double(3.0, 0.0, 50.0, 0.5)
        self.sp_fc = self._spin_double(5000.0, 0.0, 50000.0, 100.0)
        self.cb_drift = QtWidgets.QCheckBox("Drift ekle")
        self.sp_drift = self._spin_double(1.0, 0.0, 50.0, 0.5)
        g2.addWidget(QtWidgets.QLabel("Hücre L (m)"), 0, 0)
        g2.addWidget(self.sp_Lcell, 0, 1)
        g2.addWidget(QtWidgets.QLabel("Geçiş sayısı"), 0, 2)
        g2.addWidget(self.sp_passes, 0, 3)
        g2.addWidget(QtWidgets.QLabel("Baseline (V)"), 1, 0)
        g2.addWidget(self.sp_baseV, 1, 1)
        g2.addWidget(QtWidgets.QLabel("Central-burst (V)"), 1, 2)
        g2.addWidget(self.sp_burstV, 1, 3)
        g2.addWidget(QtWidgets.QLabel("Far-field (V)"), 2, 0)
        g2.addWidget(self.sp_farV, 2, 1)
        g2.addWidget(QtWidgets.QLabel("Beyaz gürültü (mV)"), 2, 2)
        g2.addWidget(self.sp_noise, 2, 3)
        g2.addWidget(QtWidgets.QLabel("Dedektör fc (Hz)"), 3, 0)
        g2.addWidget(self.sp_fc, 3, 1)
        g2.addWidget(self.cb_drift, 3, 2)
        g2.addWidget(self.sp_drift, 3, 3)

        # 3) Sayısal & FFT
        box3 = QtWidgets.QGroupBox("3) Sayısal & FFT")
        g3 = QtWidgets.QGridLayout(box3)
        self.sp_M = self._spin_int(800, 100, 30000, 100)
        self.sp_Nt = self._spin_int(6000, 1000, 300000, 500)
        self.cmb_win = QtWidgets.QComboBox()
        self.cmb_win.addItems(["hamming", "boxcar", "blackman", "gaussian"])
        self.sp_galpha = self._spin_double(3.0, 0.1, 10.0, 0.1)
        self.cmb_zf = QtWidgets.QComboBox()
        self.cmb_zf.addItems(["1", "2", "4", "8"])
        self.cmb_zf.setCurrentText("2")
        g3.addWidget(QtWidgets.QLabel("Spektral bileşen M"), 0, 0)
        g3.addWidget(self.sp_M, 0, 1)
        g3.addWidget(QtWidgets.QLabel("Zaman örnekleri Nt"), 0, 2)
        g3.addWidget(self.sp_Nt, 0, 3)
        g3.addWidget(QtWidgets.QLabel("Pencere"), 1, 0)
        g3.addWidget(self.cmb_win, 1, 1)
        g3.addWidget(QtWidgets.QLabel("Gaussian α"), 1, 2)
        g3.addWidget(self.sp_galpha, 1, 3)
        g3.addWidget(QtWidgets.QLabel("Zero-fill"), 2, 0)
        g3.addWidget(self.cmb_zf, 2, 1)
        self.cb_mertz = QtWidgets.QCheckBox("ZPD merkezle (Mertz-benzeri)")
        g3.addWidget(self.cb_mertz, 2, 2)

        # 4) Hedef Gaz & Karışım
        box4 = QtWidgets.QGroupBox("4) Hedef Gaz & Karışım")
        v4 = QtWidgets.QVBoxLayout(box4)
        row_top = QtWidgets.QHBoxLayout()
        self.cmb_target = QtWidgets.QComboBox()
        self.cmb_target.addItems(["ethanol", "CO", "CO2", "CH4", "HCN", "DMMP", "2-CEES"])
        self.cmb_bandmode = QtWidgets.QComboBox()
        self.cmb_bandmode.addItems(["auto", "all"])
        row_top.addWidget(QtWidgets.QLabel("Hedef gaz"))
        row_top.addWidget(self.cmb_target)
        row_top.addWidget(QtWidgets.QLabel("Bant seçimi"))
        row_top.addWidget(self.cmb_bandmode)
        v4.addLayout(row_top)
        grid_mix = QtWidgets.QGridLayout()
        self.mix_checks: Dict[str, QtWidgets.QCheckBox] = {}
        self.mix_scales: Dict[str, QtWidgets.QDoubleSpinBox] = {}
        gases = ["ethanol", "CO", "CO2", "CH4", "HCN", "DMMP", "2-CEES"]
        for idx, g in enumerate(gases):
            chk = QtWidgets.QCheckBox(g)
            chk.setChecked(g == "ethanol")
            sp = self._spin_double(1.0, 0.0, 20.0, 0.1)
            self.mix_checks[g] = chk
            self.mix_scales[g] = sp
            grid_mix.addWidget(chk, idx // 3, (idx % 3) * 2)
            grid_mix.addWidget(sp, idx // 3, (idx % 3) * 2 + 1)
        v4.addLayout(grid_mix)
        row_csv = QtWidgets.QHBoxLayout()
        self.btn_load_csv = QtWidgets.QPushButton("Özel absorbans CSV yükle")
        self.lbl_csv = QtWidgets.QLabel("(yok)")
        self.sp_cscale = self._spin_double(1.0, 0.0, 100.0, 0.1)
        row_csv.addWidget(self.btn_load_csv)
        row_csv.addWidget(QtWidgets.QLabel("Ölçek"))
        row_csv.addWidget(self.sp_cscale)
        v4.addLayout(row_csv)
        v4.addWidget(self.lbl_csv)
        # Görselleştirme seçenekleri (normalize + hedef overlay)
        row_vis = QtWidgets.QHBoxLayout()
        self.cb_norm = QtWidgets.QCheckBox("Normalize 0–1 (T & A)"); self.cb_norm.setChecked(True)
        self.cb_overlay = QtWidgets.QCheckBox("Hedef eğrileri arka planda"); self.cb_overlay.setChecked(True)
        self.sp_overlay_alpha = self._spin_double(0.35, 0.05, 1.0, 0.05)
        row_vis.addWidget(QtWidgets.QLabel("Görselleştirme:"))
        row_vis.addWidget(self.cb_norm)
        row_vis.addWidget(self.cb_overlay)
        row_vis.addWidget(QtWidgets.QLabel("α")); row_vis.addWidget(self.sp_overlay_alpha)
        row_vis.addStretch(1)
        v4.addLayout(row_vis)
        row_exp = QtWidgets.QHBoxLayout()
        self.btn_export_wave = QtWidgets.QPushButton("Waveform CSV kaydet")
        self.btn_export_abs = QtWidgets.QPushButton("Absorbans CSV kaydet")
        row_exp.addWidget(self.btn_export_wave)
        row_exp.addWidget(self.btn_export_abs)
        v4.addLayout(row_exp)

        # 5) Tarama / Ayna (Δx ve Animasyon)
        box5 = QtWidgets.QGroupBox("5) Tarama / Ayna (canlı)")
        g5 = QtWidgets.QGridLayout(box5)
        self.sl_dx = QtWidgets.QSlider(Qt.Horizontal)
        self.sl_dx.setMinimum(-20000)
        self.sl_dx.setMaximum(20000)
        self.sl_dx.setValue(0)
        self.sl_dx.setToolTip("Hareketli ayna statik konumu Δx [mm] (−20..+20 mm)")
        self.lbl_dx = QtWidgets.QLabel("Δx = 0.000 mm | OPD_ofs = 0.000 cm")
        self.btn_play = QtWidgets.QPushButton("▶ Oynat")
        self.btn_play.setCheckable(True)
        self.sl_anim_speed = QtWidgets.QSlider(Qt.Horizontal)
        self.sl_anim_speed.setRange(1, 100)
        self.sl_anim_speed.setValue(30)
        self.sl_anim_speed.setToolTip("Animasyon hızı")
        g5.addWidget(QtWidgets.QLabel("Δx (mm)"), 0, 0)
        g5.addWidget(self.sl_dx, 0, 1, 1, 2)
        g5.addWidget(self.lbl_dx, 1, 0, 1, 3)
        g5.addWidget(self.btn_play, 2, 0)
        g5.addWidget(QtWidgets.QLabel("Hız"), 2, 1)
        g5.addWidget(self.sl_anim_speed, 2, 2)
        self.lbl_align = QtWidgets.QLabel("Hizalama: - | tₚₑₐₖ = - ms | Δx öner: - mm")
        self.btn_autocenter = QtWidgets.QPushButton("Δx otomatik merkezle")
        g5.addWidget(self.lbl_align, 3, 0, 1, 2)
        g5.addWidget(self.btn_autocenter, 3, 2)

        # Sol panel topla
        self.ctrl_layout.addWidget(box1)
        self.ctrl_layout.addWidget(box2)
        self.ctrl_layout.addWidget(box3)
        self.ctrl_layout.addWidget(box4)
        self.ctrl_layout.addWidget(box5)
        self.ctrl_layout.addStretch(1)

        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(self.ctrl)

        # ------ Sağ: Sekmeler ------
        right = QtWidgets.QTabWidget()

        # Optik Düzen 2B
        self.layout_view = LayoutView()
        right.addTab(self.layout_view, "Optik Düzen (2B)")

        # Interferogram
        self.canvas_ifg = MplCanvas("Interferogram (V – s)")
        right.addTab(self.canvas_ifg, "Interferogram")

        # ZPD
        self.canvas_zpd = MplCanvas("ZPD Yakın (±20 ms) ve ΔV")
        right.addTab(self.canvas_zpd, "ZPD + ΔV")

        # Transmittans
        self.canvas_T = MplCanvas("Transmittans (proxy)")
        right.addTab(self.canvas_T, "Transmittans")

        # Absorbans
        self.canvas_A = MplCanvas("Absorbans (proxy)")
        right.addTab(self.canvas_A, "Absorbans")

        # Ayna/OPD
        self.canvas_mirror = MplCanvas("Ayna x(t) [cm] ve OPD=2x [cm]")
        right.addTab(self.canvas_mirror, "Ayna / OPD")

        # Özet
        self.tbl_design = QtWidgets.QTableWidget(0, 2)
        self.tbl_design.setHorizontalHeaderLabels(["Parametre", "Değer"])
        self.tbl_design.horizontalHeader().setSectionResizeMode(
            0, QtWidgets.QHeaderView.ResizeToContents
        )
        self.tbl_design.horizontalHeader().setSectionResizeMode(
            1, QtWidgets.QHeaderView.Stretch
        )
        right.addTab(self.tbl_design, "Özet")

        # Multipass Hücre
        self.cell_view = CellCanvas()
        self.cell_panel = self._build_cell_controls(self.cell_view)
        split_cell = QtWidgets.QSplitter(Qt.Horizontal)
        split_cell.addWidget(self.cell_panel)
        split_cell.addWidget(self.cell_view)
        split_cell.setStretchFactor(0, 0)
        split_cell.setStretchFactor(1, 1)
        right.addTab(split_cell, "Multipass Hücre")

        # SNR & LOD
        self.snr_panel = self._build_snr_panel()
        right.addTab(self.snr_panel, "SNR & LOD")

        # Kütüphane Eşleştirme
        self.lib_panel = self._build_library_panel()
        right.addTab(self.lib_panel, "Kütüphane Eşleştirme")

        # 3B Optik
        self.canvas_3d = MplCanvas("3B Optik Düzen (mouse ile döndür/zoom)", is3d=True)
        self._init_3d_controls(right)

        # Ana layout
        splitter = QtWidgets.QSplitter(Qt.Horizontal)
        splitter.addWidget(scroll)
        splitter.addWidget(right)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        self.setCentralWidget(splitter)

        # Durum
        self.status = self.statusBar()
        self.status.showMessage("Hazır")

        # Veri
        self.custom_abs: Optional[Tuple[np.ndarray, np.ndarray]] = None
        self.last_syn: Optional[SynthResult] = None
        self.x_static_cm: float = 0.0

        # Debounced recompute timer
        self._recompute_timer = QTimer(self)
        self._recompute_timer.setSingleShot(True)
        self._recompute_timer.timeout.connect(self._recompute_all)

        # Olaylar
        for w in [
            self.sp_lamMin,
            self.sp_lamMax,
            self.sp_res,
            self.sp_v,
            self.sp_fcol,
            self.sp_bsd,
            self.sp_bsang,
            self.sp_bsmargin,
            self.cb_override,
            self.sp_beamDia,
            self.sp_Lcell,
            self.sp_passes,
            self.sp_baseV,
            self.sp_burstV,
            self.sp_farV,
            self.sp_noise,
            self.sp_fc,
            self.cb_drift,
            self.sp_drift,
            self.sp_M,
            self.sp_Nt,
            self.cmb_win,
            self.sp_galpha,
            self.cmb_zf,
            self.cmb_target,
            self.cmb_bandmode,
            self.sp_cscale,
            self.cb_norm,
            self.cb_overlay,
            self.sp_overlay_alpha,
        ]:
            if isinstance(w, QtWidgets.QComboBox):
                w.currentIndexChanged.connect(self.schedule_recompute)
            elif isinstance(w, QtWidgets.QCheckBox):
                w.stateChanged.connect(self.schedule_recompute)
            else:
                w.valueChanged.connect(self.schedule_recompute)

        for g in self.mix_checks.values():
            g.stateChanged.connect(self.schedule_recompute)
        for s in self.mix_scales.values():
            s.valueChanged.connect(self.schedule_recompute)

        self.btn_load_csv.clicked.connect(self.load_csv_absorbance)
        self.cb_norm.stateChanged.connect(self.schedule_recompute)
        self.cb_overlay.stateChanged.connect(self.schedule_recompute)
        self.sp_overlay_alpha.valueChanged.connect(self.schedule_recompute)
        self.btn_export_wave.clicked.connect(self.export_waveform_csv)
        self.btn_export_abs.clicked.connect(self.export_absorbance_csv)

        # Tarama / Ayna
        self.sl_dx.valueChanged.connect(self._on_dx_changed)
        self.btn_play.toggled.connect(self._on_play_toggled)
        self.anim_timer = QTimer(self)
        self.anim_timer.timeout.connect(self._on_anim_tick)
        self.btn_autocenter.clicked.connect(self.autocenter_dx)
        self.cb_mertz.stateChanged.connect(self.schedule_recompute)

        # İlk çizim
        self._recompute_all()
        self.cell_recompute()
        self.snr_recompute()
        self.lib_refresh_plot()
        self._draw_3d()

    # ---------- Menü ----------
    def _build_menu(self) -> None:
        mb = self.menuBar()
        m_file = mb.addMenu("Dosya")
        act_save = QtWidgets.QAction("Preset kaydet…", self)
        act_load = QtWidgets.QAction("Preset yükle…", self)
        m_file.addAction(act_save)
        m_file.addAction(act_load)
        act_save.triggered.connect(self._save_preset)
        act_load.triggered.connect(self._load_preset)

    def _save_preset(self) -> None:
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Preset kaydet", "ftir_preset.json", "JSON (*.json)"
        )
        if not path:
            return
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(self._state_dict(), f, indent=2, ensure_ascii=False)
            self.status.showMessage(f"Preset kaydedildi: {path}", 4000)
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, "Hata", f"Kaydetme hatası: {e}")

    def _load_preset(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Preset yükle", "", "JSON (*.json)"
        )
        if not path:
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                st = json.load(f)
            self._apply_state(st)
            self.status.showMessage(f"Preset yüklendi: {path}", 4000)
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, "Hata", f"Yükleme hatası: {e}")

    def _state_dict(self) -> Dict[str, float]:
        # UI durumunu serileştir
        d: Dict[str, float] = {
            "lamMin": self.sp_lamMin.value(),
            "lamMax": self.sp_lamMax.value(),
            "res": self.sp_res.value(),
            "v": self.sp_v.value(),
            "fcol": self.sp_fcol.value(),
            "bsd": self.sp_bsd.value(),
            "bsang": self.sp_bsang.value(),
            "bsmargin": self.sp_bsmargin.value(),
            "override": 1.0 if self.cb_override.isChecked() else 0.0,
            "beamDia": self.sp_beamDia.value(),
            "Lcell": self.sp_Lcell.value(),
            "passes": float(self.sp_passes.value()),
            "baseV": self.sp_baseV.value(),
            "burstV": self.sp_burstV.value(),
            "farV": self.sp_farV.value(),
            "noise": self.sp_noise.value(),
            "fc": self.sp_fc.value(),
            "drift_on": 1.0 if self.cb_drift.isChecked() else 0.0,
            "drift": self.sp_drift.value(),
            "M": float(self.sp_M.value()),
            "Nt": float(self.sp_Nt.value()),
            "win": self.cmb_win.currentText(),
            "galpha": self.sp_galpha.value(),
            "zf": float(self.cmb_zf.currentText()),
            "target": self.cmb_target.currentText(),
            "bandmode": self.cmb_bandmode.currentText(),
            "cscale": self.sp_cscale.value(),
            "x_static_cm": self.x_static_cm,
        }
        for k, chk in self.mix_checks.items():
            d[f"mix_{k}"] = 1.0 if chk.isChecked() else 0.0
            d[f"mix_{k}_scale"] = self.mix_scales[k].value()
        return d

    def _apply_state(self, st: Dict[str, float]) -> None:
        def g(key, default):
            return st.get(key, default)

        self.sp_lamMin.setValue(g("lamMin", 2.0))
        self.sp_lamMax.setValue(g("lamMax", 14.0))
        self.sp_res.setValue(g("res", 1.0))
        self.sp_v.setValue(g("v", 1.0))
        self.sp_fcol.setValue(g("fcol", 150.0))
        self.sp_bsd.setValue(g("bsd", 12.7))
        self.sp_bsang.setValue(g("bsang", 45.0))
        self.sp_bsmargin.setValue(g("bsmargin", 0.5))
        self.cb_override.setChecked(bool(g("override", 0.0)))
        self.sp_beamDia.setValue(g("beamDia", 9.0))
        self.sp_Lcell.setValue(g("Lcell", 0.5))
        self.sp_passes.setValue(int(g("passes", 10)))
        self.sp_baseV.setValue(g("baseV", 1.2))
        self.sp_burstV.setValue(g("burstV", 0.8))
        self.sp_farV.setValue(g("farV", 0.05))
        self.sp_noise.setValue(g("noise", 3.0))
        self.sp_fc.setValue(g("fc", 5000.0))
        self.cb_drift.setChecked(bool(g("drift_on", 0.0)))
        self.sp_drift.setValue(g("drift", 1.0))
        self.sp_M.setValue(int(g("M", 800)))
        self.sp_Nt.setValue(int(g("Nt", 6000)))
        self.cmb_win.setCurrentText(str(g("win", "hamming")))
        self.sp_galpha.setValue(g("galpha", 3.0))
        self.cmb_zf.setCurrentText(str(int(g("zf", 2))))
        self.cmb_target.setCurrentText(str(g("target", "ethanol")))
        self.cmb_bandmode.setCurrentText(str(g("bandmode", "auto")))
        self.sp_cscale.setValue(g("cscale", 1.0))
        self.x_static_cm = g("x_static_cm", 0.0)
        self._update_dx_label()
        for k in ["ethanol", "CO", "CO2", "CH4", "HCN", "DMMP", "2-CEES"]:
            self.mix_checks[k].setChecked(bool(g(f"mix_{k}", 1.0 if k == "ethanol" else 0.0)))
            self.mix_scales[k].setValue(g(f"mix_{k}_scale", 1.0))
        self._recompute_all()

    # ---------- yardımcı spin ----------
    def _spin_double(self, val, vmin, vmax, step) -> QtWidgets.QDoubleSpinBox:
        sp = QtWidgets.QDoubleSpinBox()
        sp.setDecimals(6)
        sp.setRange(vmin, vmax)
        sp.setSingleStep(step)
        sp.setValue(val)
        return sp

    def _spin_int(self, val, vmin, vmax, step) -> QtWidgets.QSpinBox:
        sp = QtWidgets.QSpinBox()
        sp.setRange(vmin, vmax)
        sp.setSingleStep(step)
        sp.setValue(val)
        return sp

    # ---------- hedef referans & normalize yardımcıları ----------
    def _norm01(self, y: np.ndarray) -> np.ndarray:
        if y is None or np.size(y) == 0:
            return np.array([], dtype=float)
        y = np.asarray(y, dtype=float)
        mn = float(np.nanmin(y))
        mx = float(np.nanmax(y))
        if not np.isfinite(mn) or not np.isfinite(mx) or mx <= mn + 1e-15:
            return np.zeros_like(y)
        return np.clip((y - mn) / (mx - mn + 1e-15), 0.0, 1.0)

    def _target_reference(self, s_grid: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Seçili hedef gaz için referans A_ref ve T_ref (aynı σ ızgarasında)."""
        tgt = self.cmb_target.currentText()
        if s_grid is None or s_grid.size == 0:
            return np.array([]), np.array([])
        A_ref = gas_absorbance(s_grid, tgt)
        T_ref = np.power(10.0, -A_ref)
        return A_ref, T_ref

    def _similarity_A(self, s_grid: np.ndarray, A_meas: np.ndarray, A_ref: np.ndarray) -> float:
        """Pearson r benzerliği (Absorbans) — varsa en güçlü banda yakın ±3·FWHM pencerede."""
        if s_grid.size == 0 or A_meas.size == 0 or A_ref.size == 0:
            return float('nan')
        smin = float(s_grid.min()); smax = float(s_grid.max())
        bands = choose_target_bands(self.cmb_target.currentText(), smin, smax, mode="auto")
        if bands:
            c = bands[0]["c"]; f = max(1.0, bands[0].get("fwhm", 20.0))
            m = (s_grid >= (c - 3*f)) & (s_grid <= (c + 3*f))
        else:
            m = np.ones_like(s_grid, dtype=bool)
        if not np.any(m):
            m = np.ones_like(s_grid, dtype=bool)
        a = self._norm01(A_meas[m])
        b = self._norm01(A_ref[m])
        a = a - a.mean(); b = b - b.mean()
        denom = float(np.linalg.norm(a) * np.linalg.norm(b)) + 1e-15
        return float(np.dot(a, b) / denom)

        # ----- hizalama & faz düzeltme yardımcıları -----
    def _frac_delay(self, y: np.ndarray, dt: float, tau: float) -> np.ndarray:
        if y is None or y.size == 0:
            return y
        Y = np.fft.rfft(y)
        freqs = np.fft.rfftfreq(y.size, d=max(dt, 1e-12))
        return np.fft.irfft(Y * np.exp(-1j * 2*np.pi * freqs * tau), n=y.size)

    def _alignment_metrics(self, t: np.ndarray, I0: np.ndarray, v_cm_s: float) -> Tuple[float, float, float]:
        if t.size == 0 or I0.size == 0:
            return 0.0, 0.0, 0.0
        idx = int(np.argmax(np.abs(I0)))
        t_peak = float(t[idx])
        w = 0.010
        mL = (t >= (t_peak - w)) & (t < t_peak)
        mR = (t > t_peak) & (t <= (t_peak + w))
        eL = float(np.sum(I0[mL]**2)) + 1e-15
        eR = float(np.sum(I0[mR]**2)) + 1e-15
        asym = 1.0 - abs(eL - eR) / (eL + eR)
        x_static_cm_est = -v_cm_s * t_peak
        dx_cm = x_static_cm_est - getattr(self, 'x_static_cm', 0.0)
        return t_peak, asym, dx_cm * 10.0  # mm

    def autocenter_dx(self) -> None:
        if not self.last_syn:
            return
        syn = self.last_syn
        t_peak, _, dx_mm = self._alignment_metrics(syn.t, syn.I0, self.sp_v.value())
        new_mm = self.x_static_cm * 10.0 + dx_mm
        new_mm = max(self.sl_dx.minimum()/1000.0, min(self.sl_dx.maximum()/1000.0, new_mm))
        self.x_static_cm = new_mm / 10.0
        self.sl_dx.setValue(int(round(new_mm * 1000.0)))
        self._update_dx_label()
        self.schedule_recompute()

# =================== FTIR Hesap & Çiz (Debounced) ===================
    def schedule_recompute(self) -> None:
        # 50 ms içinde bir kez hesapla (UI akıcı kalsın)
        self._recompute_timer.start(RECOMPUTE_DEBOUNCE_MS)

    def _recompute_all(self) -> None:
        try:
            # ---- Parametreleri oku ----
            lamMin = self.sp_lamMin.value()
            lamMax = self.sp_lamMax.value()
            res = self.sp_res.value()
            v = self.sp_v.value()
            fcol = self.sp_fcol.value()
            bsd = self.sp_bsd.value()
            bsang = self.sp_bsang.value()
            bsmargin = self.sp_bsmargin.value()
            beam_override = self.cb_override.isChecked()
            beamDia = self.sp_beamDia.value()
            Lcell = self.sp_Lcell.value()
            passes = self.sp_passes.value()
            baseV = self.sp_baseV.value()
            burstV = self.sp_burstV.value()
            farV = self.sp_farV.value()
            noise = self.sp_noise.value()
            fc = self.sp_fc.value()
            drift_on = self.cb_drift.isChecked()
            drift = self.sp_drift.value()
            M = self.sp_M.value()
            Nt = self.sp_Nt.value()
            win = self.cmb_win.currentText()
            galpha = self.sp_galpha.value()
            zf = int(self.cmb_zf.currentText())
            tgt = self.cmb_target.currentText()
            mode = self.cmb_bandmode.currentText()

            # ---- Geometri ve Nyquist ----
            sigmaMax_lim = 1.0 / (lamMin * 1e-4 + 1e-20)
            OPDmax = 1.0 / max(res, 1e-12)
            stroke = OPDmax / 2.0
            thetaMax = math.sqrt(max(0.0, 1.0 / (2.0 * sigmaMax_lim * OPDmax + 1e-20)))
            fcol_cm = fcol / 10.0
            dPin = 2.0 * fcol_cm * thetaMax * 10.0
            Deff = bsd * math.cos(math.radians(bsang))
            chosen_beam = beamDia if beam_override else max(1.0, Deff - bsmargin)
            warn_fit = (chosen_beam > max(0.1, Deff - bsmargin))

            adv = nyquist_advisor(lamMin, v, Nt, res)

            # ---- Spektral ızgara ----
            sigmaMin = 1.0 / (lamMax * 1e-4 + 1e-20)
            sigmaMax = 1.0 / (lamMin * 1e-4 + 1e-20)
            M = max(10, int(M))
            sigma = np.linspace(sigmaMin, sigmaMax, M)

            # ---- Karışım absorbansı ----
            scales: Dict[str, float] = {}
            for g, chk in self.mix_checks.items():
                if chk.isChecked():
                    scales[g] = float(self.mix_scales[g].value())
            A_mix = build_mixture_absorbance(sigma, scales, Lcell, int(passes))

            # Özel CSV eklendi ise
            if self.custom_abs is not None:
                wn, Ac = self.custom_abs
                cscale = self.sp_cscale.value()
                order = np.argsort(wn)
                wn_s = wn[order]
                Ac_s = Ac[order]
                Ac_grid = np.interp(sigma, wn_s, Ac_s, left=Ac_s[0], right=Ac_s[-1])
                A_mix = A_mix + cscale * Ac_grid

            volts_opt = VoltsOptions(
                baseV=baseV,
                burstV=burstV,
                farV=farV,
                noise_mV=noise,
                fc=fc,
                drift_on=drift_on,
                drift_mV=drift,
            )

            # ---- Interferogram sentezi ----
            syn = synthesize_interferogram(
                sigma, A_mix, res, v, int(Nt), volts_opt, x_static_cm=self.x_static_cm
            )
            self.last_syn = syn

            # ---- Hizalama metrikleri & (opsiyonel) Mertz merkezleme ----
            t_peak, asym, dx_mm = self._alignment_metrics(syn.t, syn.I0, v)
            if hasattr(self, 'lbl_align'):
                self.lbl_align.setText(
                    f"Hizalama: {'iyi' if asym>0.85 else ('orta' if asym>0.6 else 'zayıf')} | tₚₑₐₖ = {1e3*t_peak:.1f} ms | Δx öner: {dx_mm:.3f} mm"
                )
            V0_in, V_in = syn.V0, syn.V
            if hasattr(self, 'cb_mertz') and self.cb_mertz.isChecked():
                dt_loc = float(syn.t[1] - syn.t[0]) if syn.t.size>1 else 1.0
                V0_in = self._frac_delay(V0_in, dt_loc, t_peak)
                V_in  = self._frac_delay(V_in,  dt_loc, t_peak)

            # ---- FFT / spektrum ----
            s_full, Tmag, Amag = spectrum_from_traces(
                syn.t, V0_in, V_in, v, win, galpha, zf
            )
            mask = (s_full >= sigmaMin) & (s_full <= sigmaMax)
            s_view = s_full[mask]
            T_view = Tmag[mask]
            A_view = Amag[mask]

            # Hedef bant(lar)
            tgbands = choose_target_bands(tgt, sigmaMin, sigmaMax, mode)

            # ---- 2B Optik çiz ----
            self.layout_view.draw_layout(chosen_beam, bsd, bsang, fcol, warn_fit)

            # ---- Interferogram grafiği ----
            ax = self.canvas_ifg.ax
            ax.clear()
            ax.plot(syn.t, syn.V0, label="I₀")
            ax.plot(syn.t, syn.V, label="I")
            ax.axvline(0.0, color="#888", ls="--", lw=1)
            ax.set_xlabel("t (s)")
            ax.set_ylabel("V")
            ax.grid(True, alpha=0.3)
            ax.legend()
            self.canvas_ifg.draw()

            # ---- ZPD + ΔV ----
            ax = self.canvas_zpd.ax
            ax.clear()
            tmin, tmax = -0.02, 0.02
            m = (syn.t >= tmin) & (syn.t <= tmax)
            ax.plot(syn.t[m], syn.V0[m], label="I₀")
            ax.plot(syn.t[m], syn.V[m], label="I")
            dv = (syn.V0 - syn.V)
            ax.plot(syn.t[m], dv[m], label="ΔV")
            if tgbands:
                b0 = tgbands[0]
                fc_line = 2.0 * v * b0["c"]
                Tc = 1.0 / max(fc_line, 1e-12)
                xv = [k * Tc for k in range(-6, 7) if tmin <= k * Tc <= tmax]
                for xx in xv:
                    ax.axvline(xx, color="#aaa", ls=":", lw=1)
            ax.set_xlim(tmin, tmax)
            ax.grid(True, alpha=0.3)
            ax.legend()
            self.canvas_zpd.draw()

            # ---- Transmittans ----
            ax = self.canvas_T.ax
            ax.clear()
            # normalize seçeneği
            if self.cb_norm.isChecked():
                T_plot = self._norm01(T_view)
                ylab = "T (0–1)"
            else:
                T_plot = T_view
                ylab = "T"
            # hedef overlay — ORİJİNAL: Transmittans grafiğinde Hedef T gösterilir
            if self.cb_overlay.isChecked():
                A_ref, T_ref = self._target_reference(s_view)
                if T_ref.size > 0:
                    T_ref_plot = self._norm01(T_ref) if self.cb_norm.isChecked() else T_ref
                    ax.plot(s_view, T_ref_plot, lw=1.0, ls='--', alpha=self.sp_overlay_alpha.value(), color="#444", label="Hedef T")
            ax.plot(s_view, T_plot, lw=1.4, label="Ölçüm T")
            if tgbands:
                for b in tgbands:
                    ax.axvline(b["c"], color="#c00", ls="--", lw=1)
            ax.set_xlabel("σ (cm⁻¹)")
            ax.set_ylabel(ylab)
            ax.grid(True, alpha=0.3)
            ax.legend()
            self.canvas_T.draw()

            # ---- Absorbans ----
            ax = self.canvas_A.ax
            ax.clear()
            if self.cb_norm.isChecked():
                A_plot = self._norm01(A_view)
                ylabA = "A (0–1)"
            else:
                A_plot = A_view
                ylabA = "A"
            # hedef overlay — ORİJİNAL: Absorbans grafiğinde Hedef A gösterilir
            if self.cb_overlay.isChecked():
                A_ref, T_ref = self._target_reference(s_view)
                if A_ref.size > 0:
                    A_ref_plot = self._norm01(A_ref) if self.cb_norm.isChecked() else A_ref
                    ax.plot(s_view, A_ref_plot, lw=1.0, ls='--', alpha=self.sp_overlay_alpha.value(), color="#666", label="Hedef A")
            ax.plot(s_view, A_plot, lw=1.4, color="#d62728", label="Ölçüm A")
            if tgbands:
                for b in tgbands:
                    ax.axvline(b["c"], color="#c00", ls="--", lw=1)
            ax.set_xlabel("σ (cm⁻¹)")
            ax.set_ylabel(ylabA)
            ax.grid(True, alpha=0.3)
            ax.legend()
            self.canvas_A.draw()

            # ---- Ayna / OPD ----
            ax = self.canvas_mirror.ax
            ax.clear()
            x = v * syn.t + self.x_static_cm
            opd = 2.0 * x
            ax.plot(syn.t, x, label="x(t) [cm]")
            ax.plot(syn.t, opd, label="OPD [cm]")
            ax.axhline(0, color="#aaa", lw=0.8)
            ax.grid(True, alpha=0.3)
            ax.legend()
            ax.set_xlabel("t (s)")
            self.canvas_mirror.draw()

            # ---- Özet tablosu ----
            # ---- Hedef benzerlik (Absorbans) ----
            try:
                A_ref_for_sim, _ = self._target_reference(s_view)
                rA = self._similarity_A(s_view, A_view, A_ref_for_sim)
            except Exception:
                rA = float('nan')

            rows = [
                ("σ_max (cm⁻¹)", f"{1.0/(lamMin*1e-4):.0f}"),
                ("Δσ hedef (cm⁻¹)", f"{res:.2f}"),
                ("OPD_max (cm)", f"{OPDmax:.3f}"),
                ("Ayna stroku s (cm)", f"{stroke:.3f}"),
                ("θ_max (mrad)", f"{thetaMax*1e3:.1f}"),
                ("OAP f_col (mm)", f"{fcol:.1f}"),
                ("Önerilen pinhole d (mm)", f"{dPin:.2f}"),
                ("BS efektif Ø (mm)", f"{Deff:.2f}"),
                ("Seçilen Beam Ø (mm)", f"{chosen_beam:.2f}"),
                ("Nyquist f_max (Hz)", f"{adv['fmax']:.0f}"),
                ("Örnekleme fs (Hz)", f"{adv['fs']:.0f}"),
                ("Öneri fs ≥ (Hz)", f"{adv['fsSuggest']:.0f}"),
                ("δOPD (cm)", f"{adv['dOPD']:.6f}"),
                ("Nyquist uygunluk", "✅" if adv["nyOK"] else "❌"),
                ("Δx statik (cm)", f"{self.x_static_cm:.4f}"),
                ("OPD ofset (cm)", f"{2.0*self.x_static_cm:.4f}"),
                ("Hedef benzerlik r(A)", f"{rA:.2f}"),
            ]
            rows.extend([
                ("t_peak (ms)", f"{1e3*t_peak:.2f}"),
                ("Hizalama asimetri", f"{asym:.2f}"),
            ])
            self.tbl_design.setRowCount(len(rows))
            for i, (k, vtxt) in enumerate(rows):
                self.tbl_design.setItem(i, 0, QtWidgets.QTableWidgetItem(k))
                self.tbl_design.setItem(i, 1, QtWidgets.QTableWidgetItem(vtxt))

            # ---- SNR paneline ΔV rms ----
            dv = (syn.V0 - syn.V)
            mz = (syn.t >= -0.01) & (syn.t <= 0.01)
            if np.any(mz):
                Vrms = float(np.sqrt(np.mean(dv[mz] ** 2)))
                self.ed_Vsig.setText(f"{Vrms:.6g}")

            # ---- Library güncelle ----
            self.lib_current_s = s_view
            self.lib_current_A = A_view
            self.lib_refresh_plot()

            # ---- 3B yeniden çiz ----
            self._draw_3d()

            # Performans uyarısı
            est_points = int(min(MAX_POINTS, (1 << int(math.ceil(math.log2(max(4, Nt * zf)))))))
            if est_points >= MAX_POINTS:
                self.status.showMessage("Uyarı: FFT boyutu güvenli üst sınıra kırpıldı.", 5000)
            else:
                self.status.showMessage("Hazır")

        except Exception as e:
            self.status.showMessage("Hata: ayrıntılar için mesaj kutusunu kontrol edin.", 4000)
            QtWidgets.QMessageBox.warning(self, "Hesaplama hatası", str(e))

    # ---------- Δx yardımcıları ----------
    def _update_dx_label(self) -> None:
        self.lbl_dx.setText(
            f"Δx = {self.x_static_cm*10.0:.3f} mm | OPD_ofs = {2.0*self.x_static_cm:.3f} cm"
        )

    def _on_dx_changed(self, val_int: int) -> None:
        dx_mm = val_int / 1000.0
        self.x_static_cm = dx_mm / 10.0  # mm → cm
        self._update_dx_label()
        self.schedule_recompute()

    # ---------- Animasyon ----------
    def _on_play_toggled(self, checked: bool) -> None:
        if checked:
            self.btn_play.setText("⏸ Duraklat")
            interval = max(10, 110 - self.sl_anim_speed.value())
            self.anim_timer.start(interval)
        else:
            self.btn_play.setText("▶ Oynat")
            self.anim_timer.stop()

    def _on_anim_tick(self) -> None:
        step = max(1, self.sl_anim_speed.value())
        val = self.sl_dx.value() + step
        if val > self.sl_dx.maximum():
            val = self.sl_dx.minimum()
        self.sl_dx.setValue(val)

    # ---------- CSV işlemleri ----------
    def load_csv_absorbance(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Absorbans CSV yükle", "", "CSV (*.csv)"
        )
        if not path:
            return
        wn: List[float] = []
        A: List[float] = []
        for enc in ("utf-8", "latin-1"):
            try:
                with open(path, "r", newline="", encoding=enc) as f:
                    rd = csv.reader(f)
                    rows = list(rd)
                break
            except Exception:
                rows = []
                continue
        if not rows:
            QtWidgets.QMessageBox.warning(self, "Hata", "CSV okunamadı.")
            return
        start = 0
        if rows and rows[0]:
            head0 = ",".join(rows[0]).lower()
            if ("wavenumber" in head0) or ("cm" in head0):
                start = 1
        for r in rows[start:]:
            if len(r) < 2:
                continue
            try:
                wn.append(float(r[0]))
                A.append(float(r[1]))
            except Exception:
                pass
        if len(wn) >= 2:
            wn_arr = np.array(wn, dtype=float)
            A_arr = np.array(A, dtype=float)
            order = np.argsort(wn_arr)
            self.custom_abs = (wn_arr[order], A_arr[order])
            self.lbl_csv.setText(f"Yüklendi: {path.split('/')[-1]} (N={len(wn)})")
        else:
            self.custom_abs = None
            self.lbl_csv.setText("(geçersiz CSV)")
        self.schedule_recompute()

    def export_waveform_csv(self) -> None:
        if not self.last_syn:
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Waveform CSV kaydet", "ftir_waveform.csv", "CSV (*.csv)"
        )
        if not path:
            return
        t = self.last_syn.t
        V0 = self.last_syn.V0
        V = self.last_syn.V
        try:
            with open(path, "w", newline="", encoding="utf-8") as f:
                wr = csv.writer(f)
                wr.writerow(["time_s", "V_background", "V_sample"])
                for i in range(t.size):
                    wr.writerow([t[i], V0[i], V[i]])
            self.status.showMessage(f"Waveform yazıldı: {path}", 4000)
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, "Hata", f"Yazma hatası: {e}")

    def export_absorbance_csv(self) -> None:
        sigma, A = self.current_absorbance_grid()
        if sigma is None:
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Absorbans CSV kaydet",
            "ftir_absorbance_mixture.csv",
            "CSV (*.csv)",
        )
        if not path:
            return
        try:
            with open(path, "w", newline="", encoding="utf-8") as f:
                wr = csv.writer(f)
                wr.writerow(["wavenumber_cm-1", "absorbance_A"])
                for i in range(sigma.size):
                    wr.writerow([sigma[i], A[i]])
            self.status.showMessage(f"Absorbans yazıldı: {path}", 4000)
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, "Hata", f"Yazma hatası: {e}")

    def current_absorbance_grid(self) -> Tuple[np.ndarray, np.ndarray]:
        lamMin = self.sp_lamMin.value()
        lamMax = self.sp_lamMax.value()
        M = self.sp_M.value()
        Lcell = self.sp_Lcell.value()
        passes = self.sp_passes.value()
        sigmaMin = 1.0 / (lamMax * 1e-4 + 1e-20)
        sigmaMax = 1.0 / (lamMin * 1e-4 + 1e-20)
        sigma = np.linspace(sigmaMin, sigmaMax, max(10, int(M)))
        scales: Dict[str, float] = {}
        for g, chk in self.mix_checks.items():
            if chk.isChecked():
                scales[g] = float(self.mix_scales[g].value())
        A_mix = build_mixture_absorbance(sigma, scales, Lcell, int(passes))
        if self.custom_abs is not None:
            wn, Ac = self.custom_abs
            cscale = self.sp_cscale.value()
            order = np.argsort(wn)
            wn_s = wn[order]
            Ac_s = Ac[order]
            Ac_grid = np.interp(sigma, wn_s, Ac_s, left=Ac_s[0], right=Ac_s[-1])
            A_mix = A_mix + cscale * Ac_grid
        return sigma, A_mix

    # ================= Multipass Hücre =================
    def _build_cell_controls(self, canvas: CellCanvas) -> QtWidgets.QWidget:
        w = QtWidgets.QWidget()
        lay = QtWidgets.QFormLayout(w)
        self.cmb_cell_type = QtWidgets.QComboBox()
        self.cmb_cell_type.addItems(["Herriott (2 ayna)", "White (3 ayna, basit)"])
        self.sp_R = self._spin_double(1000, 50, 10000, 10)
        self.sp_L = self._spin_double(500, 10, 5000, 10)
        self.sp_D = self._spin_double(100, 10, 1000, 5)
        self.sp_hole = self._spin_double(2.0, 0.2, 20.0, 0.1)
        self.sp_alpha = self._spin_double(5.0, -50.0, 50.0, 0.1)
        self.sp_beta = self._spin_double(0.0, -50.0, 50.0, 0.1)
        self.sp_x0 = self._spin_double(5.0, -50.0, 50.0, 0.1)
        self.sp_y0 = self._spin_double(5.0, -50.0, 50.0, 0.1)
        self.sp_maxb = self._spin_int(120, 10, 2000, 10)
        self.lbl_pass = QtWidgets.QLabel("Pass: -   Leff: - m")
        self.btn_cell_recalc = QtWidgets.QPushButton("Ray-Trace / Yenile")
        lay.addRow("Hücre tipi", self.cmb_cell_type)
        lay.addRow("Ayna yarıçapı R (mm)", self.sp_R)
        lay.addRow("Ayna aralığı L (mm)", self.sp_L)
        lay.addRow("Ayna çapı D (mm)", self.sp_D)
        lay.addRow("Delik yarıçapı (mm)", self.sp_hole)
        lay.addRow("α(mrad), β(mrad)", self._hbox(self.sp_alpha, self.sp_beta))
        lay.addRow("x0(mm), y0(mm)", self._hbox(self.sp_x0, self.sp_y0))
        lay.addRow("Maks. yansıma", self.sp_maxb)
        lay.addRow(self.btn_cell_recalc)
        lay.addRow(self.lbl_pass)
        self.btn_cell_recalc.clicked.connect(self.cell_recompute)
        for el in [
            self.cmb_cell_type,
            self.sp_R,
            self.sp_L,
            self.sp_D,
            self.sp_hole,
            self.sp_alpha,
            self.sp_beta,
            self.sp_x0,
            self.sp_y0,
            self.sp_maxb,
        ]:
            if isinstance(el, QtWidgets.QComboBox):
                el.currentIndexChanged.connect(self.cell_recompute)
            else:
                el.valueChanged.connect(self.cell_recompute)
        return w

    def _hbox(self, *widgets) -> QtWidgets.QWidget:
        hb = QtWidgets.QHBoxLayout()
        w = QtWidgets.QWidget()
        w.setLayout(hb)
        for wi in widgets:
            hb.addWidget(wi)
        return w

    def cell_recompute(self) -> None:
        typ = self.cmb_cell_type.currentText()
        R = self.sp_R.value()
        L = self.sp_L.value()
        D = self.sp_D.value()
        hole = self.sp_hole.value()
        a = self.sp_alpha.value()
        b = self.sp_beta.value()
        x0 = self.sp_x0.value()
        y0 = self.sp_y0.value()
        maxb = self.sp_maxb.value()
        if "Herriott" in typ:
            A, B, bounces = herriott_trace(R, L, D, hole, a, b, x0, y0, maxb)
            self.cell_view.draw_herriott(R, L, D, hole, a, b, x0, y0, maxb, A, B)
            passes = max(1, len(A) + len(B))
            Leff_m = (passes * L) / 1000.0
            self.lbl_pass.setText(f"Pass ≈ {passes}   Leff ≈ {Leff_m:.3f} m")
        else:
            spots = white_trace_simple(R, L, D, a, b, maxb)
            self.cell_view.draw_white(R, L, D, a, b, maxb, spots)
            passes = max(1, len(spots))
            Leff_m = (passes * (L * 0.9)) / 1000.0
            self.lbl_pass.setText(f"Pass ≈ {passes}   Leff ≈ {Leff_m:.3f} m")

    # ================= SNR & LOD =================
    def _build_snr_panel(self) -> QtWidgets.QWidget:
        w = QtWidgets.QWidget()
        grid = QtWidgets.QGridLayout(w)
        r = 0
        self.ed_Vsig = QtWidgets.QLineEdit("0.01")
        self.sp_NEP = self._spin_double(1e-10, 1e-14, 1e-6, 1e-12)
        self.sp_Rv = self._spin_double(1e3, 1e-1, 1e6, 1e1)
        self.sp_BW = self._spin_double(1000.0, 0.0, 1e6, 10.0)
        self.sp_Tint = self._spin_double(0.0, 0.0, 10.0, 0.01)
        self.btn_snr = QtWidgets.QPushButton("SNR Hesapla")
        self.lbl_snr = QtWidgets.QLabel("SNR: - , Vn: - V_rms")
        self.sp_Ssens = self._spin_double(1e-4, 1e-8, 1e-1, 1e-6)
        self.sp_Leff = self._spin_double(1.0, 0.01, 1000.0, 0.01)
        self.sp_Aline = self._spin_double(0.01, 1e-6, 1.0, 1e-4)
        self.sp_SNRreq = self._spin_double(3.0, 1.0, 20.0, 0.5)
        self.btn_lod = QtWidgets.QPushButton("LOD Kestir")
        self.lbl_lod = QtWidgets.QLabel("LOD: -")
        for lbl, wdg in [
            ("V_sig (rms) [V]", self.ed_Vsig),
            ("NEP [W/√Hz]", self.sp_NEP),
            ("Responsivity R_v [V/W]", self.sp_Rv),
            ("BW [Hz] (0→Tint)", self.sp_BW),
            ("Tint [s] (0→BW)", self.sp_Tint),
        ]:
            grid.addWidget(QtWidgets.QLabel(lbl), r, 0)
            grid.addWidget(wdg, r, 1)
            r += 1
        grid.addWidget(self.btn_snr, r, 0, 1, 2)
        r += 1
        grid.addWidget(self.lbl_snr, r, 0, 1, 2)
        r += 1
        grid.addWidget(self._hsep(), r, 0, 1, 2)
        r += 1
        for lbl, wdg in [
            ("S_sens [cm⁻¹ / conc]", self.sp_Ssens),
            ("L_eff [m]", self.sp_Leff),
            ("A_line", self.sp_Aline),
            ("SNR_req", self.sp_SNRreq),
        ]:
            grid.addWidget(QtWidgets.QLabel(lbl), r, 0)
            grid.addWidget(wdg, r, 1)
            r += 1
        grid.addWidget(self.btn_lod, r, 0, 1, 2)
        r += 1
        grid.addWidget(self.lbl_lod, r, 0, 1, 2)
        r += 1
        self.btn_snr.clicked.connect(self.snr_recompute)
        self.btn_lod.clicked.connect(self.lod_recompute)
        return w

    def _hsep(self) -> QtWidgets.QFrame:
        line = QtWidgets.QFrame()
        line.setFrameShape(QtWidgets.QFrame.HLine)
        line.setFrameShadow(QtWidgets.QFrame.Sunken)
        return line

    def snr_recompute(self) -> float:
        try:
            Vsig = float(self.ed_Vsig.text())
        except Exception:
            Vsig = 0.01
        NEP = self.sp_NEP.value()
        Rv = self.sp_Rv.value()
        BW = self.sp_BW.value()
        Tint = self.sp_Tint.value()
        snr, Vn = snr_lod_calc(
            Vsig, NEP, Rv, BW_Hz=(None if Tint > 0 else BW), Tint_s=(Tint if Tint > 0 else None)
        )
        self.lbl_snr.setText(f"SNR: {snr:.2f}   Vn: {Vn:.3e} V_rms")
        return float(snr)

    def lod_recompute(self) -> None:
        snr_actual = self.snr_recompute()
        Ssens = self.sp_Ssens.value()
        Leff = self.sp_Leff.value()
        Aline = self.sp_Aline.value()
        SNRreq = self.sp_SNRreq.value()
        lod = lod_from_beer(Aline, Leff, Ssens, SNRreq, snr_actual, 1.0)
        self.lbl_lod.setText(f"LOD ~ {lod:.3g} (kullanıcı birimi)")

    # ================= Kütüphane Eşleştirme =================
    def _build_library_panel(self) -> QtWidgets.QWidget:
        w = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(w)
        btns = QtWidgets.QHBoxLayout()
        self.btn_add_lib = QtWidgets.QPushButton("Kütüphane CSV ekle")
        self.btn_clear_lib = QtWidgets.QPushButton("Listeyi temizle")
        btns.addWidget(self.btn_add_lib)
        btns.addWidget(self.btn_clear_lib)
        btns.addStretch(1)
        band = QtWidgets.QHBoxLayout()
        self.sp_smin = self._spin_double(500, 0, 10000, 1)
        self.sp_smax = self._spin_double(3500, 0, 10000, 1)
        self.btn_solve = QtWidgets.QPushButton("NNLS çöz")
        band.addWidget(QtWidgets.QLabel("σ_min"))
        band.addWidget(self.sp_smin)
        band.addWidget(QtWidgets.QLabel("σ_max"))
        band.addWidget(self.sp_smax)
        band.addWidget(self.btn_solve)
        band.addStretch(1)
        self.tbl_lib = QtWidgets.QTableWidget(0, 3)
        self.tbl_lib.setHorizontalHeaderLabels(["Dosya", "Ölçek c_i", "Not"])
        self.tbl_lib.horizontalHeader().setSectionResizeMode(0, QtWidgets.QHeaderView.Stretch)
        self.tbl_lib.horizontalHeader().setSectionResizeMode(1, QtWidgets.QHeaderView.ResizeToContents)
        self.tbl_lib.horizontalHeader().setSectionResizeMode(2, QtWidgets.QHeaderView.ResizeToContents)
        self.canvas_lib = MplCanvas("Kütüphane Eşleştirme — A_meas vs A_fit")
        lay.addLayout(btns)
        lay.addLayout(band)
        lay.addWidget(self.tbl_lib)
        lay.addWidget(self.canvas_lib)
        self.btn_add_lib.clicked.connect(self.lib_add_csv)
        self.btn_clear_lib.clicked.connect(self.lib_clear)
        self.btn_solve.clicked.connect(self.lib_solve)
        self.sp_smin.valueChanged.connect(self.lib_refresh_plot)
        self.sp_smax.valueChanged.connect(self.lib_refresh_plot)
        self.lib_specs: List[Tuple[str, np.ndarray, np.ndarray]] = []
        self.lib_current_s: Optional[np.ndarray] = None
        self.lib_current_A: Optional[np.ndarray] = None
        return w

    def lib_add_csv(self) -> None:
        paths, _ = QtWidgets.QFileDialog.getOpenFileNames(
            self, "Kütüphane CSV ekle", "", "CSV (*.csv)"
        )
        for path in paths:
            wn: List[float] = []
            A: List[float] = []
            try:
                with open(path, "r", newline="", encoding="utf-8") as f:
                    rd = csv.reader(f)
                    rows = list(rd)
                if not rows:
                    continue
                start = 1 if (rows and rows[0] and ("wavenumber" in rows[0][0].lower() or "cm" in rows[0][0].lower())) else 0
                for r in rows[start:]:
                    if len(r) < 2:
                        continue
                    wn.append(float(r[0]))
                    A.append(float(r[1]))
                if len(wn) >= 2:
                    wn_arr = np.array(wn, dtype=float)
                    A_arr = np.array(A, dtype=float)
                    order = np.argsort(wn_arr)
                    self.lib_specs.append((path.split("/")[-1], wn_arr[order], A_arr[order]))
            except Exception:
                pass
        self.lib_refresh_table()
        self.lib_refresh_plot()

    def lib_clear(self) -> None:
        self.lib_specs.clear()
        self.lib_refresh_table()
        self.lib_refresh_plot()

    def lib_refresh_table(self, coeffs: Optional[np.ndarray] = None) -> None:
        self.tbl_lib.setRowCount(len(self.lib_specs))
        for i, (name, _, _) in enumerate(self.lib_specs):
            self.tbl_lib.setItem(i, 0, QtWidgets.QTableWidgetItem(name))
            self.tbl_lib.setItem(
                i, 1, QtWidgets.QTableWidgetItem("-" if coeffs is None else f"{coeffs[i]:.3g}")
            )
            self.tbl_lib.setItem(i, 2, QtWidgets.QTableWidgetItem(""))

    def lib_refresh_plot(self, coeffs: Optional[np.ndarray] = None) -> None:
        ax = self.canvas_lib.ax
        ax.clear()
        if self.lib_current_s is None or self.lib_current_A is None:
            ax.text(0.5, 0.5, "Önce FTIR simülasyonunu çalıştırın.", ha='center', va='center', transform=ax.transAxes)
            self.canvas_lib.draw()
            return
        s = self.lib_current_s
        Ameas = self.lib_current_A
        smin = min(self.sp_smin.value(), float(s.max()))
        smax = min(max(smin + 1, self.sp_smax.value()), float(s.max()))
        m = (s >= smin) & (s <= smax)
        ax.plot(s[m], Ameas[m], label="A_meas", lw=1.4, color="#d62728")
        if coeffs is not None and len(self.lib_specs) > 0:
            Afit = np.zeros_like(Ameas)
            for ci, (name, wn, A) in zip(coeffs, self.lib_specs):
                Agr = np.interp(s, wn, A, left=A[0], right=A[-1])
                Afit += max(0.0, float(ci)) * Agr
            ax.plot(s[m], Afit[m], label="A_fit", lw=1.2)
            ax.plot(s[m], (Ameas - Afit)[m], label="Residual", lw=1.0, ls='--')
        ax.set_xlim(smin, smax)
        ax.set_xlabel("σ (cm⁻¹)")
        ax.set_ylabel("Absorbans")
        ax.grid(True, alpha=0.3)
        ax.legend()
        self.canvas_lib.draw()

    def lib_solve(self) -> None:
        if self.lib_current_s is None or self.lib_current_A is None or len(self.lib_specs) == 0:
            return
        s = self.lib_current_s
        Ameas = self.lib_current_A
        smin = self.sp_smin.value()
        smax = self.sp_smax.value()
        m = (s >= smin) & (s <= smax)
        s_seg = s[m]
        y = Ameas[m]
        if s_seg.size < 3:
            return
        Acols: List[np.ndarray] = []
        for name, wn, A in self.lib_specs:
            Agr = np.interp(s_seg, wn, A, left=A[0], right=A[-1])
            Acols.append(Agr)
        A_mat = np.vstack(Acols).T
        coeffs = nnls_pg(A_mat, y, max_iter=4000, tol=1e-7)
        self.lib_refresh_table(coeffs)
        self.lib_refresh_plot(coeffs)

    # ================= 3B Optik =================
    def _init_3d_controls(self, tabwidget: QtWidgets.QTabWidget) -> None:
        panel = QtWidgets.QWidget()
        grid = QtWidgets.QGridLayout(panel)
        r = 0
        # Geometri parametreleri (mm; basit yerleşim)
        self.sp3_src = self._spin_double(0.0, -500.0, 500.0, 1.0)  # kaynak x
        self.sp3_oapx = self._spin_double(100.0, -1000.0, 1000.0, 1.0)
        self.sp3_bs_x = self._spin_double(200.0, -1000.0, 1000.0, 1.0)
        self.sp3_bs_y = self._spin_double(0.0, -500.0, 500.0, 1.0)
        self.sp3_bs_z = self._spin_double(0.0, -500.0, 500.0, 1.0)
        self.sp3_bs_yaw = self._spin_double(45.0, 0.0, 90.0, 1.0)
        self.sp3_mfix_x = self._spin_double(450.0, -1000.0, 2000.0, 1.0)
        self.sp3_mmov_y = self._spin_double(250.0, -1000.0, 2000.0, 1.0)
        self.sp3_det_x = self._spin_double(80.0, -1000.0, 1000.0, 1.0)
        self.sp3_oap_tilt = self._spin_double(0.0, -20.0, 20.0, 0.5)
        for lbl, wdg in [
            ("Kaynak X (mm)", self.sp3_src),
            ("OAP X (mm)", self.sp3_oapx),
            ("OAP Tilt Z (°)", self.sp3_oap_tilt),
            ("BS X/Y/Z (mm)", self._hbox(self.sp3_bs_x, self.sp3_bs_y, self.sp3_bs_z)),
            ("BS yaw (°)", self.sp3_bs_yaw),
            ("Sabit ayna X (mm)", self.sp3_mfix_x),
            ("Hareketli ayna Y (mm)", self.sp3_mmov_y),
            ("Dedektör X (mm)", self.sp3_det_x),
        ]:
            grid.addWidget(QtWidgets.QLabel(lbl), r, 0)
            grid.addWidget(wdg, r, 1)
            r += 1
        help_lbl = QtWidgets.QLabel("""
            Mouse: 3B eksenleri döndür/zoom (Matplotlib 3D).
            Kaydırıcı/alanları değiştir → ışın yolları ve optik elemanlar canlı güncellenir.
            Not: 3B ray-trace eğitim amaçlıdır; düzlemsel BS ve düz ayna yansımalarını içerir.
        """)
        help_lbl.setWordWrap(True)
        grid.addWidget(help_lbl, r, 0, 1, 2)
        r += 1
        split = QtWidgets.QSplitter(Qt.Horizontal)
        split.addWidget(panel)
        split.addWidget(self.canvas_3d)
        split.setStretchFactor(0, 0)
        split.setStretchFactor(1, 1)
        tabwidget.addTab(split, "3B Optik")
        for el in [
            self.sp3_src,
            self.sp3_oapx,
            self.sp3_bs_x,
            self.sp3_bs_y,
            self.sp3_bs_z,
            self.sp3_bs_yaw,
            self.sp3_mfix_x,
            self.sp3_mmov_y,
            self.sp3_det_x,
            self.sp3_oap_tilt,
        ]:
            el.valueChanged.connect(self._draw_3d)

    def _draw_3d(self) -> None:
        ax = self.canvas_3d.ax
        ax.clear()
        src_x = self.sp3_src.value()
        oap_x = self.sp3_oapx.value()
        bs_pos = np.array(
            [self.sp3_bs_x.value(), self.sp3_bs_y.value(), self.sp3_bs_z.value()],
            dtype=float,
        )
        bs_yaw = self.sp3_bs_yaw.value()
        mfix_pos = np.array([self.sp3_mfix_x.value(), 0.0, 0.0], dtype=float)
        mmov_pos = np.array([self.sp3_bs_x.value(), self.sp3_mmov_y.value(), 0.0], dtype=float)
        det_pos = np.array([self.sp3_det_x.value(), 0.0, 0.0], dtype=float)
        # oap_tilt kullanılmıyor (simgesel)

        def draw_plane(center, normal, size=40, color="#cccccc", alpha=0.4):
            n = normal / (np.linalg.norm(normal) + 1e-12)
            tmp = np.array([1, 0, 0], dtype=float) if abs(n[0]) < 0.9 else np.array([0, 1, 0], dtype=float)
            u = np.cross(n, tmp)
            u = u / (np.linalg.norm(u) + 1e-12)
            v = np.cross(n, u)
            s = size
            pts = [
                center + s * (+u + v),
                center + s * (+u - v),
                center + s * (-u - v),
                center + s * (-u + v),
            ]
            X = [p[0] for p in pts] + [pts[0][0]]
            Y = [p[1] for p in pts] + [pts[0][1]]
            Z = [p[2] for p in pts] + [pts[0][2]]
            ax.plot(X, Y, Z, color=color, alpha=alpha)

        # OAP
        ax.scatter([oap_x], [0], [0], c='b', s=30, label='OAP')
        # BS düzlemi (~45° yaw)
        n_bs = rot_z(bs_yaw).dot(np.array([1 / np.sqrt(2), 0, 1 / np.sqrt(2)]))
        draw_plane(bs_pos, n_bs, size=35, color='#999999', alpha=0.5)
        ax.text(bs_pos[0], bs_pos[1], bs_pos[2], "BS")
        # Aynalar
        n_mfix = np.array([-1.0, 0, 0])
        draw_plane(mfix_pos, n_mfix, size=30, color='#777777', alpha=0.5)
        ax.text(mfix_pos[0], 0, 0, "Sabit")
        n_mmov = np.array([0, -1.0, 0])
        draw_plane(mmov_pos, n_mmov, size=30, color='#777777', alpha=0.5)
        ax.text(mmov_pos[0], mmov_pos[1], 0, "Hareketli")
        # Dedektör & Kaynak
        ax.scatter([det_pos[0]], [det_pos[1]], [det_pos[2]], c='g', s=30, label='Detektör')
        src = np.array([src_x, 0, 0], dtype=float)
        ax.scatter([src[0]], [0], [0], c='r', s=30, label='Kaynak')

        # Işın izi (yaklaşık)
        ray_color1 = "#1f77b4"
        ray_color2 = "#d62728"
        v0 = np.array([1, 0, 0], dtype=float)

        def trace_branch(to_mirror_center, mirror_normal, col):
            p = src
            v = v0
            p1 = line_plane_intersect(p, v, bs_pos, n_bs)
            if p1 is None:
                return
            segment(ax, p, p1, color=col, lw=1.6)
            v = reflect(v, n_bs)
            p2 = line_plane_intersect(p1, v, to_mirror_center, mirror_normal)
            if p2 is None:
                return
            segment(ax, p1, p2, color=col, lw=1.6)
            v = reflect(v, mirror_normal)
            p3 = line_plane_intersect(p2, v, bs_pos, n_bs)
            if p3 is None:
                return
            segment(ax, p2, p3, color=col, lw=1.6)
            v = reflect(v, n_bs)
            p4 = det_pos
            segment(ax, p3, p4, color=col, lw=1.6)

        trace_branch(mfix_pos, n_mfix, ray_color1)
        trace_branch(mmov_pos, n_mmov, ray_color2)

        ax.set_xlabel("X (mm)")
        ax.set_ylabel("Y (mm)")
        ax.set_zlabel("Z (mm)")
        ax.set_box_aspect([1, 1, 0.6])
        lim = max(200, abs(self.sp3_mfix_x.value()) + 100, abs(self.sp3_mmov_y.value()) + 100)
        ax.set_xlim(src_x - 50, max(self.sp3_mfix_x.value() + 100, self.sp3_bs_x.value() + 100))
        ax.set_ylim(-lim, lim)
        ax.set_zlim(-lim / 2, lim / 2)
        ax.grid(True, alpha=0.2)
        ax.legend(loc='upper left')
        self.canvas_3d.draw()


# =========================== Uygulama ===========================

def main() -> None:
    app = QtWidgets.QApplication(sys.argv)
    win = FTIRWindow()
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
