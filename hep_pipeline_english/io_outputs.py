"""Output writing utilities.

This module saves NumPy arrays, metadata JSON, and small preview text files.
"""

import os
import json
from datetime import datetime

import numpy as np

from .config_constants import FEATURE_NAMES, ALGO_CODE_TO_NAME

def save_datasets(workflow, all_datasets, n_events):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    saved_npy_paths = []

    run_dir = os.path.join(workflow.work_dir, f"run_{workflow.process_name}_{ts}")
    os.makedirs(run_dir, exist_ok=True)

    print("\n[4/4] GUARDANDO DATASETS")
    print(f"  Carpeta: {run_dir}")
    print("-" * 50)

    for cfg_key, payload in all_datasets.items():
        data = payload["data"]
        algo = payload["algo"]
        algo_code = int(payload["algo_code"])
        R = payload["R"]
        event_figures = payload.get("event_figures", [])

        cfg_dir = os.path.join(run_dir, cfg_key)
        os.makedirs(cfg_dir, exist_ok=True)

        base = os.path.join(cfg_dir, f"jets_{workflow.process_name}_{cfg_key}")

        npy_path = f"{base}.npy"
        np.save(npy_path, data)
        print(f"   {cfg_key}/jets_...npy  ({data.shape[0]} jets)")
        saved_npy_paths.append(npy_path)

        meta = {
            "timestamp": ts,
            "process": workflow.process_name,
            "n_events_accepted": int(n_events),
            "n_jets": int(data.shape[0]),
            "algorithm": algo,
            "algorithm_code": algo_code,
            "algorithm_code_map": {str(k): v for k, v in ALGO_CODE_TO_NAME.items()},
            "R": float(R),
            "jet_pt_min": float(workflow.jet_pt_min),
            "min_hard_parton_pt_proxy": float(workflow.min_hard_parton_pt),
            "n_features": int(N_FEATURES),
            "features": {str(i): name for i, name in enumerate(FEATURE_NAMES)},
            "feature_notes": {
                "flavour": "PDG ID CON SIGNO (antiparticulas negativas) como float32",
                "reco": "Parametric smearing applied to GEN variables",
                "btag_ctag": "Proxies using B and C hadrons plus flavour matching",
                "fractions": "NEF/NHF/CEF/CHF desde constituyentes finales de Pythia8",
                "jetR": "Column 22 with the jet radius R used for that jet",
                "algoCode": "Column 23 with the algorithm code (1=antikt, 2=kt, 3=cambridge)",
            },
            "notes": {
                "visible_jets": "Neutrinos excluidos del clustering",
                "flavour_matching": "Matching jet-parton con status abs in {23,33,43,51,52,53,59,62} y cono adaptativo",
                "hardness_proxy": "NO equivale exactamente a PhaseSpace:pTHatMin de Pythia-only; es max pT de parton relevante",
                "flowSim_hint": "If FlowSim requires strict integers for flavour and algoCode, cast columns 4 and 23 to int when loading",
            },
        }

        with open(f"{base}_metadata.json", "w") as f:
            json.dump(meta, f, indent=2)

        with open(f"{base}_preview.txt", "w") as f:
            f.write(f"# {cfg_key} | {data.shape[0]} jets | {n_events} eventos aceptados\n")
            f.write(f"# jet_pt_min = {workflow.jet_pt_min} GeV\n")
            f.write(f"# min_hard_parton_pt_proxy = {workflow.min_hard_parton_pt} GeV (0=off)\n")
            f.write(f"# algoCode map: 1=antikt, 2=kt, 3=cambridge\n")
            f.write("# " + "  ".join(f"{i}:{n}" for i, n in enumerate(FEATURE_NAMES)) + "\n")

            if data.shape[0] > 0:
                np.savetxt(f, data[:10], fmt="%10.4f")
            else:
                f.write("# Dataset vacio\n")

        if workflow.save_jet_figures:
            fig_dir = os.path.join(cfg_dir, "figures")
            workflow._plot_global_dataset_figures(data, cfg_key, fig_dir)

            ev_fig_dir = os.path.join(cfg_dir, "event_figures")
            for item in event_figures:
                workflow._plot_event_jets_eta_phi_from_arrays(
                    jets_eta_phi_pt=item["jets_eta_phi_pt"],
                    cfg_key=cfg_key,
                    source_event_idx=item["source_event_idx"],
                    accepted_event_idx=item["accepted_event_idx"],
                    out_dir=ev_fig_dir
                )

    readme_path = os.path.join(run_dir, "README.txt")
    with open(readme_path, "w") as f:
        f.write(f"Run: {workflow.process_name}\n")
        f.write(f"Fecha: {ts}\n")
        f.write(f"Eventos aceptados: {n_events}\n")
        f.write(f"Jet pT minimo: {workflow.jet_pt_min} GeV\n")
        f.write(f"Hardness proxy (max pT parton): {workflow.min_hard_parton_pt} GeV (0=off)\n\n")

        f.write("Configuraciones generadas:\n")
        for cfg_key, payload in all_datasets.items():
            f.write(
                f"  {cfg_key:25s} to {payload['data'].shape[0]} jets"
                f"   (algo={payload['algo']}, code={payload['algo_code']}, R={payload['R']})\n"
            )

        f.write("\nEstructura:\n")
        f.write(f"  run_{workflow.process_name}_{ts}/\n")
        for cfg_key in all_datasets:
            f.write(f"    {cfg_key}/\n")
            f.write("      jets_...npy\n")
            f.write("      jets_..._metadata.json\n")
            f.write("      jets_..._preview.txt\n")
            if workflow.save_jet_figures:
                f.write("      figures/\n")
                f.write("        hist_pt.png\n")
                f.write("        hist_eta.png\n")
                f.write("        hist_phi.png\n")
                f.write("        scatter_eta_phi_ptcolor.png\n")
                f.write("      event_figures/\n")
                f.write("        srcEv_..._accEv_..._eta_phi.png\n")
        if workflow.save_feynman_diagrams:
            f.write("    feynman_diagrams/\n")
            f.write("      (copias y conversiones de diagramas encontrados en MG5)\n")

        f.write("\nColumnas del .npy:\n")
        for i, name in enumerate(FEATURE_NAMES):
            f.write(f"  [{i:2d}] {name}\n")

        f.write("\nMapa de algoritmo (algoCode):\n")
        for code, name in sorted(ALGO_CODE_TO_NAME.items()):
            f.write(f"  {code} to {name}\n")

        f.write("\nImportant note for FlowSim:\n")
        f.write("  Column [4] flavour stores the signed PDG ID (float32).\n")
        f.write("  Column [23] algoCode is also stored as float32.\n")
        f.write("  If your pipeline expects integers, convert those columns to int when loading.\n")

    if workflow.save_feynman_diagrams:
        workflow._collect_feynman_diagrams(run_dir)

    print(f"\n  README to {readme_path}")
    return run_dir, saved_npy_paths

