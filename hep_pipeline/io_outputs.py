"""IO de outputs: .npy, metadata, preview, README y carpetas de salida."""

import os
import json
from .config_constants import N_FEATURES, FEATURE_NAMES
from datetime import datetime

import numpy as np

from .config_constants import FEATURE_NAMES, ALGO_CODE_TO_NAME

def save_datasets(workflow, all_datasets, n_events):
    # ↑ Guarda todos los datasets y artefactos del run en una carpeta timestamp.
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    # ↑ Timestamp para nombre único y ordenable cronológicamente.
    saved_npy_paths = []
    # ↑ Lista para reportar rutas de .npy al final.

    run_dir = os.path.join(workflow.work_dir, f"run_{workflow.process_name}_{ts}")
    # ↑ Carpeta principal del run (ej. run_ttbar_had_4j_20260225_123456).
    os.makedirs(run_dir, exist_ok=True)
    # ↑ Crea la carpeta si no existe.

    print("\n[4/4] GUARDANDO DATASETS")
    # ↑ Título del paso 4.
    print(f"  Carpeta: {run_dir}")
    # ↑ Muestra la ruta donde se guardará todo.
    print("-" * 50)
    # ↑ Separador visual.

    for cfg_key, payload in all_datasets.items():
        # ↑ Recorre cada configuración (antikt_R0.4, etc.).
        data = payload["data"]
        # ↑ Array [N,24] de esa configuración.
        algo = payload["algo"]
        # ↑ Nombre del algoritmo.
        algo_code = int(payload["algo_code"])
        # ↑ Código numérico del algoritmo.
        R = payload["R"]
        # ↑ Radio.
        event_figures = payload.get("event_figures", [])
        # ↑ Info mínima para figuras por evento (si existe).

        cfg_dir = os.path.join(run_dir, cfg_key)
        # ↑ Carpeta específica de la configuración.
        os.makedirs(cfg_dir, exist_ok=True)
        # ↑ La crea si no existe.

        base = os.path.join(cfg_dir, f"jets_{workflow.process_name}_{cfg_key}")
        # ↑ Prefijo base para archivos (npy, metadata, preview).

        # .npy principal
        # ↑ Archivo principal de datos para ML/análisis.
        npy_path = f"{base}.npy"
        # ↑ Ruta completa del .npy.
        np.save(npy_path, data)
        # ↑ Guarda el array NumPy en disco.
        print(f"  ✓ {cfg_key}/jets_...npy  ({data.shape[0]} jets)")
        # ↑ Log de guardado.
        saved_npy_paths.append(npy_path)
        # ↑ Acumula la ruta para resumen final.

        # metadata
        # ↑ Diccionario JSON con contexto, mapeos y notas de interpretación.
        meta = {
            "timestamp": ts,
            # ↑ Timestamp del run.
            "process": workflow.process_name,
            # ↑ Nombre del proceso MG5.
            "n_events_accepted": int(n_events),
            # ↑ Número de eventos aceptados (tras filtros).
            "n_jets": int(data.shape[0]),
            # ↑ Número de jets en este dataset.
            "algorithm": algo,
            # ↑ Nombre del algoritmo FastJet.
            "algorithm_code": algo_code,
            # ↑ Código numérico del algoritmo.
            "algorithm_code_map": {str(k): v for k, v in ALGO_CODE_TO_NAME.items()},
            # ↑ Mapa código->nombre serializable como JSON (claves string).
            "R": float(R),
            # ↑ Radio del jet.
            "jet_pt_min": float(workflow.jet_pt_min),
            # ↑ Corte de pT aplicado.
            "min_hard_parton_pt_proxy": float(workflow.min_hard_parton_pt),
            # ↑ Umbral del proxy de dureza.
            "n_features": int(N_FEATURES),
            # ↑ Número de columnas/features.
            "features": {str(i): name for i, name in enumerate(FEATURE_NAMES)},
            # ↑ Mapa índice->nombre de columna.
            "feature_notes": {
                "flavour": "PDG ID CON SIGNO (antipartículas negativas) como float32",
                # ↑ Aclara el formato de flavour.
                "reco": "Smearing parametrico sobre variables GEN",
                # ↑ Aclara que reco es toy smearing.
                "btag_ctag": "Proxies usando hadrones B/C y flavour matching",
                # ↑ Aclara lógica de taggers.
                "fractions": "NEF/NHF/CEF/CHF desde constituyentes finales de Pythia8",
                # ↑ Aclara origen de fracciones.
                "jetR": "Columna 22 con el valor de R usado para ese jet",
                # ↑ Aclara feature 22.
                "algoCode": "Columna 23 con código del algoritmo (1=antikt,2=kt,3=cambridge)",
                # ↑ Aclara feature 23.
            },
            "notes": {
                "visible_jets": "Neutrinos excluidos del clustering",
                # ↑ Nota física importante para interpretar jets.
                "flavour_matching": "Matching jet-parton con status abs in {23,33,43,51,52,53,59,62} y cono adaptativo",
                # ↑ Resume la estrategia de flavour matching.
                "hardness_proxy": "NO equivale exactamente a PhaseSpace:pTHatMin de Pythia-only; es max pT de partón relevante",
                # ↑ Aclara limitación del filtro de dureza.
                "flowSim_hint": "Si FlowSim quiere enteros estrictos para flavour/algoCode, castear columnas 4 y 23 a int al cargar",
                # ↑ Consejo práctico para tu pipeline de ML.
            },
        }

        with open(f"{base}_metadata.json", "w") as f:
            # ↑ Abre archivo metadata para escritura.
            json.dump(meta, f, indent=2)
            # ↑ Guarda JSON con indentación legible.

        # preview txt
        # ↑ Archivo de vista rápida para inspección humana sin cargar NumPy.
        with open(f"{base}_preview.txt", "w") as f:
            # ↑ Abre preview en texto plano.
            f.write(f"# {cfg_key} | {data.shape[0]} jets | {n_events} eventos aceptados\n")
            # ↑ Header con resumen del dataset.
            f.write(f"# jet_pt_min = {workflow.jet_pt_min} GeV\n")
            # ↑ Guarda corte de pT.
            f.write(f"# min_hard_parton_pt_proxy = {workflow.min_hard_parton_pt} GeV (0=off)\n")
            # ↑ Guarda filtro de dureza.
            f.write(f"# algoCode map: 1=antikt, 2=kt, 3=cambridge\n")
            # ↑ Guarda mapa de códigos.
            f.write("# " + "  ".join(f"{i}:{n}" for i, n in enumerate(FEATURE_NAMES)) + "\n")
            # ↑ Línea con mapeo de columnas para lectura rápida.

            if data.shape[0] > 0:
                # ↑ Si hay jets, imprime primeras 10 filas.
                np.savetxt(f, data[:10], fmt="%10.4f")
                # ↑ Guarda primeras filas formateadas.
            else:
                # ↑ Si dataset vacío, se deja aviso explícito.
                f.write("# Dataset vacío\n")

        # Figuras globales + por evento
        # ↑ Se generan solo si el usuario dejó activado save_jet_figures.
        if workflow.save_jet_figures:
            fig_dir = os.path.join(cfg_dir, "figures")
            # ↑ Carpeta para histogramas/scatter globales.
            workflow._plot_global_dataset_figures(data, cfg_key, fig_dir)
            # ↑ Genera las figuras globales (incluyendo phi corregido a [-pi, pi]).

            ev_fig_dir = os.path.join(cfg_dir, "event_figures")
            # ↑ Carpeta para scatters por evento.
            for item in event_figures:
                # ↑ Recorre eventos seleccionados para figuras.
                workflow._plot_event_jets_eta_phi_from_arrays(
                    jets_eta_phi_pt=item["jets_eta_phi_pt"],
                    # ↑ Jets del evento (eta,phi,pt).
                    cfg_key=cfg_key,
                    # ↑ Nombre de config para título.
                    source_event_idx=item["source_event_idx"],
                    # ↑ Índice original de Pythia.
                    accepted_event_idx=item["accepted_event_idx"],
                    # ↑ Índice de aceptado.
                    out_dir=ev_fig_dir
                    # ↑ Carpeta de salida.
                )

    # README general
    # ↑ Archivo resumen del run completo (todas las configs).
    readme_path = os.path.join(run_dir, "README.txt")
    # ↑ Ruta del README.
    with open(readme_path, "w") as f:
        # ↑ Abre README para escritura.
        f.write(f"Run: {workflow.process_name}\n")
        # ↑ Nombre del proceso.
        f.write(f"Fecha: {ts}\n")
        # ↑ Timestamp.
        f.write(f"Eventos aceptados: {n_events}\n")
        # ↑ Eventos aceptados.
        f.write(f"Jet pT mínimo: {workflow.jet_pt_min} GeV\n")
        # ↑ Corte pT.
        f.write(f"Hardness proxy (max pT partón): {workflow.min_hard_parton_pt} GeV (0=off)\n\n")
        # ↑ Filtro de dureza.

        f.write("Configuraciones generadas:\n")
        # ↑ Encabezado listado de configs.
        for cfg_key, payload in all_datasets.items():
            # ↑ Recorre cada config y resume jets y parámetros.
            f.write(
                f"  {cfg_key:25s} -> {payload['data'].shape[0]} jets"
                f"   (algo={payload['algo']}, code={payload['algo_code']}, R={payload['R']})\n"
            )
            # ↑ Una línea por config.

        f.write("\nEstructura:\n")
        # ↑ Sección que describe árbol de carpetas generado.
        f.write(f"  run_{workflow.process_name}_{ts}/\n")
        # ↑ Carpeta raíz del run.
        for cfg_key in all_datasets:
            # ↑ Por cada configuración...
            f.write(f"    {cfg_key}/\n")
            # ↑ Carpeta de config.
            f.write("      jets_...npy\n")
            # ↑ Archivo de datos.
            f.write("      jets_..._metadata.json\n")
            # ↑ Metadata.
            f.write("      jets_..._preview.txt\n")
            # ↑ Preview.
            if workflow.save_jet_figures:
                # ↑ Solo describe figuras si se generaron.
                f.write("      figures/\n")
                # ↑ Carpeta de figuras globales.
                f.write("        hist_pt.png\n")
                # ↑ Histograma pT.
                f.write("        hist_eta.png\n")
                # ↑ Histograma eta.
                f.write("        hist_phi.png\n")
                # ↑ Histograma phi (en [-pi, pi]).
                f.write("        scatter_eta_phi_ptcolor.png\n")
                # ↑ Scatter global.
                f.write("      event_figures/\n")
                # ↑ Carpeta de figuras por evento.
                f.write("        srcEv_..._accEv_..._eta_phi.png\n")
                # ↑ Nombre patrón de figuras por evento.
        if workflow.save_feynman_diagrams:
            # ↑ Solo describe carpeta de diagramas si estaba activa la opción.
            f.write("    feynman_diagrams/\n")
            # ↑ Carpeta de diagramas.
            f.write("      (copias y conversiones de diagramas encontrados en MG5)\n")
            # ↑ Nota descriptiva.

        f.write("\nColumnas del .npy:\n")
        # ↑ Sección con mapeo de columnas.
        for i, name in enumerate(FEATURE_NAMES):
            # ↑ Recorre nombres de features.
            f.write(f"  [{i:2d}] {name}\n")
            # ↑ Una línea por columna.

        f.write("\nMapa de algoritmo (algoCode):\n")
        # ↑ Sección con mapa de códigos de algoritmo.
        for code, name in sorted(ALGO_CODE_TO_NAME.items()):
            # ↑ Ordena por código.
            f.write(f"  {code} -> {name}\n")
            # ↑ Escribe el mapeo.

        f.write("\nNota importante para FlowSim:\n")
        # ↑ Nota específica para tu workflow de ML.
        f.write("  La columna [4] 'flavour' guarda el PDG ID con signo (float32).\n")
        # ↑ Recuerda que flavour está en float32.
        f.write("  La columna [23] 'algoCode' también está en float32.\n")
        # ↑ Idem para algoCode.
        f.write("  Si tu pipeline espera enteros, convierte esas columnas a int al cargar.\n")
        # ↑ Recomendación práctica.

    # Diagramas de Feynman
    # ↑ Paso opcional al final para recopilar diagramas MG5.
    if workflow.save_feynman_diagrams:
        workflow._collect_feynman_diagrams(run_dir)
        # ↑ Busca/copia/convierte diagramas dentro del run.

    print(f"\n  README -> {readme_path}")
    # ↑ Muestra ruta al README generado.
    return run_dir, saved_npy_paths
    # ↑ Devuelve la carpeta del run y las rutas de los .npy guardados.

# --------------------------
# Diagnóstico opcional
# --------------------------
# ↑ Método de ayuda para debug si flavour sale muy frecuentemente en 0.
