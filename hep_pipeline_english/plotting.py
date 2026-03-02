"""Plotting utilities (Matplotlib) for the HEP workflow.

Separated from the core logic to keep the main pipeline code shorter.
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .config_constants import FEATURE_NAMES

def plot_event_jets_eta_phi_from_arrays(jets_eta_phi_pt, cfg_key, source_event_idx, accepted_event_idx, out_dir):
    """
    jets_eta_phi_pt: lista de tuplas (eta, phi, pt)
    """

    if not jets_eta_phi_pt:
        return

    etas = np.array([x[0] for x in jets_eta_phi_pt], dtype=float)

    phis = np.array([wrap_phi(x[1]) for x in jets_eta_phi_pt], dtype=float)

    pts = np.array([x[2] for x in jets_eta_phi_pt], dtype=float)

    plt.figure(figsize=(7, 5))

    sc = plt.scatter(
        etas, phis,
        c=pts,
        s=40 + 2.5 * np.clip(pts, 0, 200),
        alpha=0.85
    )

    plt.xlabel(r"$\eta$")
    plt.ylabel(r"$\phi$")

    plt.ylim(-np.pi, np.pi)

    plt.title(f"{cfg_key} | srcEv={source_event_idx} | accEv={accepted_event_idx} | color=pT")

    plt.grid(True, alpha=0.25)

    cbar = plt.colorbar(sc)
    cbar.set_label(r"$p_T$ [GeV]")

    plt.tight_layout()

    os.makedirs(out_dir, exist_ok=True)

    out_path = os.path.join(out_dir, f"srcEv_{source_event_idx:06d}_accEv_{accepted_event_idx:06d}_eta_phi.png")

    plt.savefig(out_path, dpi=140)
    plt.close()

def plot_global_dataset_figures(workflow, dataset, cfg_key, out_dir):
    """
    dataset shape [N,24]
    Uses columns:
      0 pt_gen, 1 eta_gen, 2 phi_gen
    """

    if dataset.shape[0] == 0:
        return

    os.makedirs(out_dir, exist_ok=True)

    pt = dataset[:, 0]
    eta = dataset[:, 1]

    phi = np.array([wrap_phi(x) for x in dataset[:, 2]], dtype=float)

    plt.figure(figsize=(6, 4))
    plt.hist(pt, bins=60, alpha=0.85)
    plt.xlabel(r"$p_T$ [GeV]")
    plt.ylabel("Jets")
    plt.title(f"{cfg_key} | pT")
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "hist_pt.png"), dpi=140)
    plt.close()

    plt.figure(figsize=(6, 4))
    plt.hist(eta, bins=60, alpha=0.85)
    plt.xlabel(r"$\eta$")
    plt.ylabel("Jets")
    plt.title(f"{cfg_key} | eta")
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "hist_eta.png"), dpi=140)
    plt.close()

    plt.figure(figsize=(6, 4))
    plt.hist(
        phi,
        bins=60,
        range=(-np.pi, np.pi),
        alpha=0.85
    )
    plt.xlim(-np.pi, np.pi)
    plt.xlabel(r"$\phi$")
    plt.ylabel("Jets")
    plt.title(f"{cfg_key} | phi")
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "hist_phi.png"), dpi=140)
    plt.close()

    if dataset.shape[0] > workflow.max_scatter_points_global:
        idx = workflow.rng.choice(dataset.shape[0], size=workflow.max_scatter_points_global, replace=False)
        eta_s, phi_s, pt_s = eta[idx], phi[idx], pt[idx]
    else:
        eta_s, phi_s, pt_s = eta, phi, pt

    plt.figure(figsize=(7, 5))
    sc = plt.scatter(eta_s, phi_s, c=pt_s, s=8, alpha=0.65)
    plt.xlabel(r"$\eta$")
    plt.ylabel(r"$\phi$")

    plt.ylim(-np.pi, np.pi)

    plt.title(f"{cfg_key} | eta vs phi (color = pT)")
    plt.grid(True, alpha=0.25)
    cbar = plt.colorbar(sc)
    cbar.set_label(r"$p_T$ [GeV]")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "scatter_eta_phi_ptcolor.png"), dpi=140)
    plt.close()

