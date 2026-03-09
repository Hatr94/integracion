"""Helpers físicos y utilidades pequeñas (sin estado)."""

import numpy as np

# ==========================
# Helpers globales
# ==========================
# ↑ Funciones auxiliares reutilizables en varias partes del workflow.

def wrap_phi(phi: float) -> float:
    # ↑ Define una función para normalizar el ángulo azimutal phi a un rango estándar.
    """Envuelve ángulo phi al rango (-pi, pi]."""
    # ↑ Docstring corta: este convenio es muy común en HEP para evitar saltos 0↔2π.

    while phi > np.pi:
        # ↑ Si phi es mayor que +π, lo bajamos restando 2π hasta entrar al rango.
        phi -= 2.0 * np.pi

    while phi <= -np.pi:
        # ↑ Si phi es menor o igual a -π, lo subimos sumando 2π.
        phi += 2.0 * np.pi

    return float(phi)
    # ↑ Se retorna como float nativo (no NumPy scalar) por limpieza y compatibilidad.


def jet_quality_id(fracs: dict) -> int:
    # ↑ Calcula un JetID tipo CMS (proxy simplificado) usando fracciones y multiplicidades.
    """
    JetID estilo CMS loose/tight basado en fracciones de energía.
    Retorna: 0=fail, 1=loose, 3=tight
    """
    # ↑ Documenta la lógica de salida: 0 falla, 1 loose, 3 tight (estilo común de codificación).

    chf = fracs["chf"]
    # ↑ Fracción hadrónica cargada.
    nhf = fracs["nhf"]
    # ↑ Fracción hadrónica neutra.
    nef = fracs["nef"]
    # ↑ Fracción electromagnética neutra.
    nc = fracs["ncharged"]
    # ↑ Número de constituyentes cargados.

    loose = (nhf < 0.99) and (nef < 0.99) and (nc > 0)
    # ↑ Criterio loose: evita jets extremadamente neutros/sospechosos y exige algo cargado.

    tight = loose and (nhf < 0.90) and (nef < 0.90) and (chf > 0.0)
    # ↑ Criterio tight: además restringe más NHF/NEF y exige CHF positiva.

    if tight:
        # ↑ Si pasa tight, se devuelve primero porque es la categoría más estricta.
        return 3

    if loose:
        # ↑ Si no pasa tight pero sí loose, devuelve código 1.
        return 1

    return 0
    # ↑ Si no cumple loose, falla JetID.


def quark_gluon_likelihood(fracs: dict) -> float:
    # ↑ Construye un proxy simple de quark/gluon likelihood basado en topología del jet.
    """
    Proxy simple de QGL:
      - más constituyentes -> más gluon-like
      - más CHF -> más quark-like
    """
    # ↑ Idea física simplificada: jets de gluón tienden a ser más "anchos" y con más constituyentes.

    nc = min(fracs["n_const"], 60)
    # ↑ Recorta n_const a 60 para evitar que jets extremos dominen la escala.

    qgl = 1.0 - 0.5 * (nc / 60.0) + 0.5 * fracs["chf"]
    # ↑ Construcción lineal simple:
    #   - más constituyentes baja el score (más gluon-like),
    #   - más CHF lo sube (más quark-like).

    return float(np.clip(qgl, 0.0, 1.0))
    # ↑ Se recorta al rango [0,1] y se devuelve float estándar.


