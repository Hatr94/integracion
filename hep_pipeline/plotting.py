"""Funciones de plotting (Matplotlib) para el workflow HEP.

Se separa para mantener el core más corto.
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from .utils_physics import wrap_phi

from .config_constants import FEATURE_NAMES

def plot_event_jets_eta_phi_from_arrays(jets_eta_phi_pt, cfg_key, source_event_idx, accepted_event_idx, out_dir):
    # ↑ Grafica jets de un solo evento en el plano (eta, phi), coloreados por pT.
    """
    jets_eta_phi_pt: lista de tuplas (eta, phi, pt)
    """
    # ↑ Entrada mínima necesaria para graficar un evento sin depender del dataset completo.

    if not jets_eta_phi_pt:
        # ↑ Si la lista está vacía, no se genera figura.
        return

    etas = np.array([x[0] for x in jets_eta_phi_pt], dtype=float)
    # ↑ Extrae eta de cada jet y lo convierte a array NumPy.

    phis = np.array([wrap_phi(x[1]) for x in jets_eta_phi_pt], dtype=float)
    # ↑ CORRECCIÓN:
    #   Envuelve phi explícitamente a (-pi, pi] para consistencia visual y física.

    pts = np.array([x[2] for x in jets_eta_phi_pt], dtype=float)
    # ↑ Extrae pT de cada jet para usarlo como color y tamaño.

    plt.figure(figsize=(7, 5))
    # ↑ Crea figura de tamaño razonable para scatter.

    sc = plt.scatter(
        etas, phis,
        # ↑ Coordenadas X=eta, Y=phi.
        c=pts,
        # ↑ Color de cada punto codifica pT.
        s=40 + 2.5 * np.clip(pts, 0, 200),
        # ↑ Tamaño de marcador aumenta con pT, con saturación a 200 GeV.
        alpha=0.85
        # ↑ Transparencia para mejor visibilidad cuando hay puntos cercanos.
    )

    plt.xlabel(r"$\eta$")
    # ↑ Etiqueta del eje X.
    plt.ylabel(r"$\phi$")
    # ↑ Etiqueta del eje Y.

    plt.ylim(-np.pi, np.pi)
    # ↑ CORRECCIÓN:
    #   Fija el rango de phi en el eje Y a [-pi, pi] (evita visualizaciones 0..2pi).

    plt.title(f"{cfg_key} | srcEv={source_event_idx} | accEv={accepted_event_idx} | color=pT")
    # ↑ Título con configuración + índice del evento original y aceptado.

    plt.grid(True, alpha=0.25)
    # ↑ Rejilla suave para leer mejor la geometría.

    cbar = plt.colorbar(sc)
    # ↑ Agrega barra de color asociada al scatter.
    cbar.set_label(r"$p_T$ [GeV]")
    # ↑ Etiqueta de la barra de color.

    plt.tight_layout()
    # ↑ Ajusta márgenes para que no se recorten etiquetas/título.

    os.makedirs(out_dir, exist_ok=True)
    # ↑ Crea la carpeta de salida si no existe.

    out_path = os.path.join(out_dir, f"srcEv_{source_event_idx:06d}_accEv_{accepted_event_idx:06d}_eta_phi.png")
    # ↑ Nombre de archivo con índices formateados para orden correcto.

    plt.savefig(out_path, dpi=140)
    # ↑ Guarda la imagen en PNG con resolución moderada.
    plt.close()
    # ↑ Cierra la figura para liberar memoria (muy importante en loops grandes).

def plot_global_dataset_figures(workflow, dataset, cfg_key, out_dir):
    # ↑ Genera histogramas globales y scatter de todo el dataset de una configuración.
    """
    dataset shape [N,24]
    Usa columnas:
      0 pt_gen, 1 eta_gen, 2 phi_gen
    """
    # ↑ Se trabaja con las variables GEN almacenadas en las primeras columnas.

    if dataset.shape[0] == 0:
        # ↑ Si no hay jets, no hay nada que graficar.
        return

    os.makedirs(out_dir, exist_ok=True)
    # ↑ Crea carpeta de figuras globales si hace falta.

    pt = dataset[:, 0]
    # ↑ Extrae pT gen.
    eta = dataset[:, 1]
    # ↑ Extrae eta gen.

    phi = np.array([wrap_phi(x) for x in dataset[:, 2]], dtype=float)
    # ↑ CORRECCIÓN CLAVE:
    #   Se envuelve phi explícitamente a (-pi, pi] para asegurar histogramas/scatters consistentes.

    # Hist pT
    # ↑ Bloque para histograma de pT.
    plt.figure(figsize=(6, 4))
    # ↑ Nueva figura.
    plt.hist(pt, bins=60, alpha=0.85)
    # ↑ Histograma de pT con 60 bins.
    plt.xlabel(r"$p_T$ [GeV]")
    # ↑ Eje X: pT.
    plt.ylabel("Jets")
    # ↑ Eje Y: conteo de jets.
    plt.title(f"{cfg_key} | pT")
    # ↑ Título de la figura.
    plt.grid(True, alpha=0.25)
    # ↑ Rejilla.
    plt.tight_layout()
    # ↑ Ajuste de layout.
    plt.savefig(os.path.join(out_dir, "hist_pt.png"), dpi=140)
    # ↑ Guarda la imagen.
    plt.close()
    # ↑ Cierra figura.

    # Hist eta
    # ↑ Bloque para histograma de eta.
    plt.figure(figsize=(6, 4))
    # ↑ Nueva figura independiente.
    plt.hist(eta, bins=60, alpha=0.85)
    # ↑ Histograma de pseudorapidez.
    plt.xlabel(r"$\eta$")
    # ↑ Eje X: eta.
    plt.ylabel("Jets")
    # ↑ Eje Y: conteo.
    plt.title(f"{cfg_key} | eta")
    # ↑ Título.
    plt.grid(True, alpha=0.25)
    # ↑ Rejilla.
    plt.tight_layout()
    # ↑ Ajuste de layout.
    plt.savefig(os.path.join(out_dir, "hist_eta.png"), dpi=140)
    # ↑ Guarda imagen.
    plt.close()
    # ↑ Cierra figura.

    # Hist phi (forzado a [-pi, pi])
    # ↑ Bloque corregido para histograma de phi con rango físico estándar.
    plt.figure(figsize=(6, 4))
    # ↑ Nueva figura para phi.
    plt.hist(
        phi,
        # ↑ Datos de phi ya envueltos.
        bins=60,
        # ↑ Número de bins.
        range=(-np.pi, np.pi),
        # ↑ CORRECCIÓN:
        #   Fuerza el rango del histograma a [-pi, pi] (no 0..2pi).
        alpha=0.85
        # ↑ Transparencia.
    )
    plt.xlim(-np.pi, np.pi)
    # ↑ Refuerza visualmente el eje X al mismo rango exacto.
    plt.xlabel(r"$\phi$")
    # ↑ Eje X: phi.
    plt.ylabel("Jets")
    # ↑ Eje Y: conteo.
    plt.title(f"{cfg_key} | phi")
    # ↑ Título.
    plt.grid(True, alpha=0.25)
    # ↑ Rejilla.
    plt.tight_layout()
    # ↑ Ajuste de layout.
    plt.savefig(os.path.join(out_dir, "hist_phi.png"), dpi=140)
    # ↑ Guarda histograma corregido.
    plt.close()
    # ↑ Cierra figura.

    # Scatter eta-phi con color pT (sampleado si hay demasiados puntos)
    # ↑ Scatter global del dataset en plano η-φ, con color = pT.
    if dataset.shape[0] > workflow.max_scatter_points_global:
        # ↑ Si hay demasiados jets, se muestrea para no saturar RAM/imagen.
        idx = workflow.rng.choice(dataset.shape[0], size=workflow.max_scatter_points_global, replace=False)
        # ↑ Elige índices aleatorios sin reemplazo.
        eta_s, phi_s, pt_s = eta[idx], phi[idx], pt[idx]
        # ↑ Toma submuestra de variables.
    else:
        # ↑ Si hay pocos puntos, usa el dataset completo.
        eta_s, phi_s, pt_s = eta, phi, pt
        # ↑ No se muestrea.

    plt.figure(figsize=(7, 5))
    # ↑ Figura ligeramente más grande para scatter.
    sc = plt.scatter(eta_s, phi_s, c=pt_s, s=8, alpha=0.65)
    # ↑ Scatter: X=eta, Y=phi, color por pT, tamaño fijo pequeño, algo transparente.
    plt.xlabel(r"$\eta$")
    # ↑ Etiqueta X.
    plt.ylabel(r"$\phi$")
    # ↑ Etiqueta Y.

    plt.ylim(-np.pi, np.pi)
    # ↑ CORRECCIÓN:
    #   Fija el rango de phi (eje Y) a [-pi, pi].

    plt.title(f"{cfg_key} | eta vs phi (color = pT)")
    # ↑ Título del scatter.
    plt.grid(True, alpha=0.25)
    # ↑ Rejilla tenue.
    cbar = plt.colorbar(sc)
    # ↑ Barra de color.
    cbar.set_label(r"$p_T$ [GeV]")
    # ↑ Etiqueta de la barra de color.
    plt.tight_layout()
    # ↑ Ajuste de layout.
    plt.savefig(os.path.join(out_dir, "scatter_eta_phi_ptcolor.png"), dpi=140)
    # ↑ Guarda scatter global.
    plt.close()
    # ↑ Cierra figura.

# --------------------------
# Diagramas de Feynman
# --------------------------
# ↑ Métodos para recopilar y convertir diagramas producidos por MG5.
