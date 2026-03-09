# hep_pipeline/puppi.py
"""Implementación simple de PUPPI (PileUp Per Particle Identification).

Esta versión está hecha para integrarse con tu pipeline:
MadGraph -> Pythia8 -> FastJet.

Entrada esperada:
    particles = [(PseudoJet, pid, is_charged, is_pu), ...]

Salida:
    lista de PseudoJet "limpios" (4-momento reescalado por w_i y filtrado).

Notas:
- Para df=1, la CDF de chi-cuadrado se puede escribir como:
    F(x;1) = erf(sqrt(x/2))
  así evitamos depender de scipy.
- Usamos vecinos = charged LV (is_charged & ~is_pu), que corresponde al caso "central"
  con tracking ideal (muy común en estudios MC).
"""

from __future__ import annotations

from math import log, sqrt, erf
import numpy as np
import fastjet as fj


def _wrap_dphi(dphi: float) -> float:
    """Envuelve Δφ a (-π, π]."""
    while dphi > np.pi:
        dphi -= 2.0 * np.pi
    while dphi <= -np.pi:
        dphi += 2.0 * np.pi
    return dphi


def _deltaR(eta1: float, phi1: float, eta2: float, phi2: float) -> float:
    """Distancia ΔR en (η,φ)."""
    return sqrt((eta1 - eta2) ** 2 + _wrap_dphi(phi1 - phi2) ** 2)


def _left_rms(x: np.ndarray, med: float) -> float:
    """RMS usando SOLO la parte izquierda (x < med)."""
    left = x[x < med]
    if left.size == 0:
        return 1e-6
    return float(np.sqrt(np.mean((left - med) ** 2)))


def _compute_alpha(
    etas: np.ndarray,
    phis: np.ndarray,
    pts: np.ndarray,
    i: int,
    neighbor_mask: np.ndarray,
    R0: float = 0.3,
    Rmin: float = 0.02,
) -> float:
    """Calcula α_i = log( sum_j pT_j / ΔR_ij ), con Rmin <= ΔR <= R0."""
    eta_i, phi_i = float(etas[i]), float(phis[i])
    acc = 0.0
    for j in np.where(neighbor_mask)[0]:
        if j == i:
            continue
        dr = _deltaR(eta_i, phi_i, float(etas[j]), float(phis[j]))
        if dr < Rmin or dr > R0:
            continue
        acc += float(pts[j]) / dr
    if acc <= 0.0:
        return -np.inf
    return log(acc)


def _chi2_cdf_df1(x: float) -> float:
    """CDF de χ² con 1 grado de libertad: F(x;1) = erf(sqrt(x/2))."""
    if x <= 0.0:
        return 0.0
    return float(erf(sqrt(x / 2.0)))


def puppi_clean_pseudojets(
    particles,
    *,
    R0: float = 0.3,
    Rmin: float = 0.02,
    wcut: float = 0.1,
    ptcut: float = 0.2,
):
    """Aplica PUPPI y devuelve pseudojets reescalados/filtrados.

    Parámetros típicos (paper):
      - R0 = 0.3
      - Rmin = 0.02
      - wcut = 0.1
      - ptcut = 0.1 + 0.007*NPU
    """
    n = len(particles)
    if n == 0:
        return []

    pjs = [t[0] for t in particles]
    is_ch = np.array([bool(t[2]) for t in particles], dtype=bool)
    is_pu = np.array([bool(t[3]) for t in particles], dtype=bool)

    pts = np.array([pj.pt() for pj in pjs], dtype=float)
    etas = np.array([pj.eta() for pj in pjs], dtype=float)
    phis = np.array([pj.phi() for pj in pjs], dtype=float)

    # Charged LV y charged PU (según overlay)
    ch_lv = is_ch & (~is_pu)
    ch_pu = is_ch & is_pu

    # 1) Calibración del evento: alphas de charged-PU usando vecinos charged-LV
    alpha_pu = []
    for i in np.where(ch_pu)[0]:
        a = _compute_alpha(etas, phis, pts, i, neighbor_mask=ch_lv, R0=R0, Rmin=Rmin)
        if np.isfinite(a):
            alpha_pu.append(a)
    alpha_pu = np.array(alpha_pu, dtype=float)

    # Si no hay estadística suficiente, no hacemos limpieza (fallback)
    if alpha_pu.size < 10:
        return pjs

    a_med = float(np.median(alpha_pu))
    a_sig = _left_rms(alpha_pu, a_med)
    if not np.isfinite(a_sig) or a_sig <= 0.0:
        return pjs

    # 2) Pesos w_i
    w = np.zeros(n, dtype=float)

    for i in range(n):
        # charged PU -> 0 ; charged LV -> 1
        if is_ch[i] and is_pu[i]:
            w[i] = 0.0
            continue
        if is_ch[i] and (not is_pu[i]):
            w[i] = 1.0
            continue

        # neutrals: usa vecinos charged-LV
        a_i = _compute_alpha(etas, phis, pts, i, neighbor_mask=ch_lv, R0=R0, Rmin=Rmin)
        if (not np.isfinite(a_i)) or (a_i <= a_med):
            w[i] = 0.0
        else:
            chi2_i = ((a_i - a_med) / a_sig) ** 2
            w[i] = _chi2_cdf_df1(float(chi2_i))

    # 3) Cortes
    w[(w < wcut) | ((w * pts) < ptcut)] = 0.0

    # 4) Reescalar 4-vectores
    out = []
    for i, pj in enumerate(pjs):
        if w[i] <= 0.0:
            continue
        newpj = fj.PseudoJet(w[i] * pj.px(), w[i] * pj.py(), w[i] * pj.pz(), w[i] * pj.e())
        newpj.set_user_index(pj.user_index())
        out.append(newpj)

    return out
