#!/usr/bin/env python3
"""
Workflow HEP completo: MadGraph -> Pythia8 -> FastJet
Genera datasets de jets con 24 features (GEN + RECO proxies + config del jet).

Salida por configuración (algoritmo, R):
  - .npy              datos principales [N_jets, 24]
  - _metadata.json    metadatos y mapeo de columnas
  - _preview.txt      preview de los primeros 10 jets
  - figures/          histogramas y scatter eta-phi (color = pT)
  - event_figures/    scatter eta-phi por evento (muestra limitada)

Además:
  - feynman_diagrams/ copia de diagramas detectados en la carpeta de MG5 (si existen)
    y conversión automática de .ps/.eps a .pdf/.jpg (si están disponibles ps2pdf + magick)

Columnas (24):
  0  pt_gen
  1  eta_gen
  2  phi_gen
  3  m_gen
  4  flavour          (PDG ID con signo; antipartículas negativas)
  5  btag
  6  recoPt
  7  recoPhi
  8  recoEta
  9  muon_pT
 10  recoNConst
 11  nef
 12  nhf
 13  cef
 14  chf
 15  qgl
 16  jetId
 17  ncharged
 18  nneutral
 19  ctag
 20  nSV
 21  recoMass
 22  jetR             (R usado para ese jet)
 23  algoCode         (1=antikt, 2=kt, 3=cambridge)
"""

import os
import glob
import gzip
import json
import shutil
import subprocess
from datetime import datetime

import numpy as np
import matplotlib
matplotlib.use("Agg")  # backend sin GUI (útil en servidores)
import matplotlib.pyplot as plt

import pythia8
import fastjet as fj


# ==========================
# Constantes físicas / PDG IDs
# ==========================

FEATURE_NAMES = [
    "pt_gen", "eta_gen", "phi_gen", "m_gen", "flavour",
    "btag", "recoPt", "recoPhi", "recoEta", "muon_pT",
    "recoNConst", "nef", "nhf", "cef", "chf",
    "qgl", "jetId", "ncharged", "nneutral", "ctag",
    "nSV", "recoMass", "jetR", "algoCode"
]

N_FEATURES = len(FEATURE_NAMES)

# Mapeo numérico de algoritmo (para guardar en columna algoCode)
ALGO_NAME_TO_CODE = {
    "antikt": 1,
    "kt": 2,
    "cambridge": 3,
}
ALGO_CODE_TO_NAME = {v: k for k, v in ALGO_NAME_TO_CODE.items()}

# Hadrones con quark b (proxy b-tag)
B_HADRON_IDS = {
    511, 521, 531, 541, 5122, 5132, 5232, 5332,
    -511, -521, -531, -541, -5122, -5132, -5232, -5332
}

# Hadrones con quark c (proxy c-tag)
C_HADRON_IDS = {
    411, 421, 431, 4122, 4132, 4232, 4332,
    -411, -421, -431, -4122, -4132, -4232, -4332
}

# Partículas de vida larga (proxy de vértices secundarios)
LONG_LIVED_IDS_ABS = {
    310, 130, 3122, 3112, 3222, 3312, 3334, 421, 411
}

# Partones candidatos para flavour matching
QUARK_GLUON_IDS_ABS = {1, 2, 3, 4, 5, 6, 21}

# Status relevantes de Pythia para partones "finales" antes de hadronización / resonancias
RELEVANT_STATUS_ABS = {23, 33, 43, 51, 52, 53, 59, 62}

# Invisibles típicos (no entran en jets visibles)
NEUTRINO_IDS_ABS = {12, 14, 16}


# ==========================
# Helpers globales
# ==========================

def wrap_phi(phi: float) -> float:
    """Envuelve ángulo phi al rango (-pi, pi]."""
    while phi > np.pi:
        phi -= 2.0 * np.pi
    while phi <= -np.pi:
        phi += 2.0 * np.pi
    return float(phi)


def jet_quality_id(fracs: dict) -> int:
    """
    JetID estilo CMS loose/tight basado en fracciones de energía.
    Retorna: 0=fail, 1=loose, 3=tight
    """
    chf = fracs["chf"]
    nhf = fracs["nhf"]
    nef = fracs["nef"]
    nc = fracs["ncharged"]

    loose = (nhf < 0.99) and (nef < 0.99) and (nc > 0)
    tight = loose and (nhf < 0.90) and (nef < 0.90) and (chf > 0.0)

    if tight:
        return 3
    if loose:
        return 1
    return 0


def quark_gluon_likelihood(fracs: dict) -> float:
    """
    Proxy simple de QGL:
      - más constituyentes -> más gluon-like
      - más CHF -> más quark-like
    """
    nc = min(fracs["n_const"], 60)
    qgl = 1.0 - 0.5 * (nc / 60.0) + 0.5 * fracs["chf"]
    return float(np.clip(qgl, 0.0, 1.0))


# ==========================
# Workflow principal
# ==========================

class HEPWorkflow:
    def __init__(self):
        # Puedes sobrescribir la ruta con variable de entorno MG5_PATH si quieres
        self.mg5_path = os.environ.get(
            "MG5_PATH",
            os.path.expanduser("~/HEP/MG5_aMC_v3_7_0/bin/mg5_aMC")
        )

        self.work_dir = os.getcwd()
        self.lhe_file = None
        self.output_dir = None
        self.process_name = None

        # Configuración por defecto EDITABLE: (R -> algoritmo)
        self.jet_configs = [
            {"R": 0.4, "algo_name": "antikt",    "algo_func": fj.antikt_algorithm,    "algo_code": ALGO_NAME_TO_CODE["antikt"]},
            {"R": 0.8, "algo_name": "kt",        "algo_func": fj.kt_algorithm,        "algo_code": ALGO_NAME_TO_CODE["kt"]},
            {"R": 1.0, "algo_name": "cambridge", "algo_func": fj.cambridge_algorithm, "algo_code": ALGO_NAME_TO_CODE["cambridge"]},
        ]

        self.n_events = 1000
        self.jet_pt_min = 15.0   # corte de jets reconstruidos (editable)

        # Proxy de "dureza" del evento (NO es exactamente pTHatMin de Pythia-only)
        self.min_hard_parton_pt = 0.0  # GeV; 0 = desactivado

        # Figuras y outputs extra
        self.save_feynman_diagrams = True
        self.save_jet_figures = True
        self.max_event_figures_per_cfg = 30
        self.max_scatter_points_global = 20000

        # RNG reproducible
        self.rng = np.random.default_rng(42)
        self.pythia_seed = 42

    # --------------------------
    # UI
    # --------------------------
    def print_header(self):
        print("\n" + "=" * 80)
        print("  WORKFLOW HEP: MadGraph -> Pythia8 -> FastJet (features físicas)")
        print("=" * 80)
        print(f"  Directorio de trabajo: {self.work_dir}")
        print("=" * 80 + "\n")

    def _parse_jet_configs_string(self, txt: str):
        """
        Formato esperado:
          0.4:antikt,0.8:kt,1.0:cambridge

        Algoritmos válidos:
          antikt, kt, cambridge
        """
        algo_map = {
            "antikt": fj.antikt_algorithm,
            "kt": fj.kt_algorithm,
            "cambridge": fj.cambridge_algorithm,
            "ca": fj.cambridge_algorithm,
            "cam": fj.cambridge_algorithm,
        }

        out = []
        for item in txt.split(","):
            item = item.strip()
            if not item:
                continue

            if ":" not in item:
                raise ValueError(f"Falta ':' en '{item}'. Usa formato R:algoritmo")

            r_str, a_str = item.split(":", 1)
            R = float(r_str.strip())
            if R <= 0:
                raise ValueError(f"R inválido ({R}). Debe ser > 0.")

            algo_key = a_str.strip().lower()
            if algo_key not in algo_map:
                raise ValueError(
                    f"Algoritmo inválido '{algo_key}'. Usa antikt, kt o cambridge."
                )

            algo_name = "cambridge" if algo_key in {"ca", "cam"} else algo_key

            out.append({
                "R": float(R),
                "algo_name": algo_name,
                "algo_func": algo_map[algo_key],
                "algo_code": ALGO_NAME_TO_CODE[algo_name],
            })

        if not out:
            raise ValueError("No se pudo parsear ninguna configuración de jets.")

        return out

    def run_madgraph_interactive(self):
        print("[1/4] MADGRAPH INTERACTIVO")
        print("-" * 50)
        print("  • Genera tu proceso normalmente")
        print("  • Usa 'output NOMBRE' para el directorio de salida")
        print("  • Al terminar escribe 'quit'\n")

        if not os.path.isfile(self.mg5_path):
            raise FileNotFoundError(
                f"No encuentro mg5_aMC en:\n  {self.mg5_path}\n"
                "Edita self.mg5_path o exporta MG5_PATH con la ruta correcta."
            )

        input("  Presiona ENTER para abrir MadGraph...")
        subprocess.run([self.mg5_path], check=False)

        raw_name = input("\nNombre del directorio de salida que usaste: ").strip().rstrip("/")
        if not raw_name:
            raise ValueError("No ingresaste el nombre del directorio de salida de MadGraph.")

        self.process_name = os.path.basename(raw_name)

        patterns = [
            os.path.join(self.work_dir, self.process_name, "Events", "run_01", "unweighted_events.lhe.gz"),
            os.path.join(self.work_dir, self.process_name, "Events", "run_01", "unweighted_events.lhe"),
            os.path.join(self.work_dir, "**", self.process_name, "**", "unweighted_events.lhe.gz"),
            os.path.join(self.work_dir, "**", self.process_name, "**", "unweighted_events.lhe"),
        ]

        for pat in patterns:
            hits = glob.glob(pat, recursive=True)
            if hits:
                hits = sorted(hits, key=os.path.getmtime, reverse=True)
                self.lhe_file = hits[0]
                break

        if self.lhe_file:
            print(f"  ✓ Archivo LHE encontrado: {self.lhe_file}")
            self.output_dir = os.path.dirname(self.lhe_file)
        else:
            self.lhe_file = input("  Ruta completa al .lhe o .lhe.gz: ").strip()
            if not self.lhe_file:
                raise ValueError("No se proporcionó ruta al archivo .lhe/.lhe.gz.")
            if not os.path.exists(self.lhe_file):
                raise FileNotFoundError(f"No existe: {self.lhe_file}")
            self.output_dir = os.path.dirname(self.lhe_file)

    def configure_analysis(self) -> bool:
        print("\n[2/4] CONFIGURACIÓN")
        print("-" * 50)

        n_in = input(f"  Eventos a procesar (intentos Pythia) (default {self.n_events}): ").strip()
        if n_in:
            self.n_events = int(n_in)
            if self.n_events <= 0:
                raise ValueError("El número de eventos debe ser > 0.")

        pt_in = input(f"  pT mínimo de jets [GeV] (default {self.jet_pt_min}): ").strip()
        if pt_in:
            self.jet_pt_min = float(pt_in)
            if self.jet_pt_min < 0:
                raise ValueError("El pT mínimo debe ser >= 0.")

        hard_in = input(
            f"  Corte proxy de dureza del evento (max pT partón) [GeV] (default {self.min_hard_parton_pt}, 0=off): "
        ).strip()
        if hard_in:
            self.min_hard_parton_pt = float(hard_in)
            if self.min_hard_parton_pt < 0:
                raise ValueError("El corte proxy de dureza debe ser >= 0.")

        print("\n  Configuración de jets por R (editable)")
        print("    Formato: R:algoritmo,R:algoritmo,...")
        print("    Algoritmos válidos: antikt, kt, cambridge")
        print("    Ejemplo: 0.4:antikt,0.8:kt,1.0:cambridge")

        default_cfg_str = ",".join(f'{cfg["R"]}:{cfg["algo_name"]}' for cfg in self.jet_configs)
        cfg_in = input(f"  Configs jets (default {default_cfg_str}): ").strip()
        if cfg_in:
            self.jet_configs = self._parse_jet_configs_string(cfg_in)

        figs_in = input("  ¿Guardar figuras de jets (globales + por evento)? (s/n, default s): ").strip().lower()
        if figs_in in {"n", "no"}:
            self.save_jet_figures = False

        diag_in = input("  ¿Copiar/convertir diagramas de Feynman de MG5 si existen? (s/n, default s): ").strip().lower()
        if diag_in in {"n", "no"}:
            self.save_feynman_diagrams = False

        if self.save_jet_figures:
            nevfig_in = input(
                f"  Máx. figuras por evento por configuración (default {self.max_event_figures_per_cfg}): "
            ).strip()
            if nevfig_in:
                self.max_event_figures_per_cfg = int(nevfig_in)
                if self.max_event_figures_per_cfg < 0:
                    raise ValueError("max_event_figures_per_cfg debe ser >= 0.")

        print("\n  Resumen:")
        print(f"    Eventos (intentos) : {self.n_events}")
        print(f"    pT min jets        : {self.jet_pt_min} GeV")
        print(f"    Hardness proxy     : {self.min_hard_parton_pt} GeV (0=off)")
        print(f"    Figuras jets       : {self.save_jet_figures}")
        print(f"    Diagramas MG5      : {self.save_feynman_diagrams}")
        print("    Jet configs        :")
        for cfg in self.jet_configs:
            print(f'      R={cfg["R"]}  ->  {cfg["algo_name"]} (code={cfg["algo_code"]})')

        ans = input("\n  ¿Continuar? (s/n): ").strip().lower()
        return ans in {"s", "si", "sí", "y", "yes"}

    # --------------------------
    # Procesamiento
    # --------------------------
    def _decompress_lhe_if_needed(self):
        if self.lhe_file.endswith(".gz"):
            out_path = self.lhe_file[:-3]
            if os.path.exists(out_path):
                print(f"  LHE ya descomprimido, usando: {out_path}")
            else:
                print(f"  Descomprimiendo LHE -> {out_path}")
                with gzip.open(self.lhe_file, "rb") as fi, open(out_path, "wb") as fo:
                    shutil.copyfileobj(fi, fo)
            self.lhe_file = out_path

    def _init_pythia(self):
        p = pythia8.Pythia()
        p.readString("Beams:frameType = 4")
        p.readString(f"Beams:LHEF = {self.lhe_file}")
        p.readString("PartonLevel:MPI = on")
        p.readString("PartonLevel:ISR = on")
        p.readString("PartonLevel:FSR = on")
        p.readString("HadronLevel:Hadronize = on")

        # Semilla reproducible de Pythia
        p.readString("Random:setSeed = on")
        p.readString(f"Random:seed = {self.pythia_seed}")

        p.readString("Print:quiet = on")

        if not p.init():
            raise RuntimeError("Pythia8 no pudo inicializarse con el archivo LHE dado.")
        return p

    def _extract_partons_for_matching(self, pythia):
        """
        Extrae partones relevantes (quarks/gluón) evitando duplicados en cadenas
        de shower/decaimiento (si tiene hija del mismo sabor, se omite).

        Devuelve tuplas: (pid, eta, phi, pt)
        """
        partons = []

        for i in range(pythia.event.size()):
            p = pythia.event[i]

            pid_abs = abs(p.id())
            if pid_abs not in QUARK_GLUON_IDS_ABS:
                continue

            stat_abs = abs(p.status())
            if stat_abs not in RELEVANT_STATUS_ABS:
                continue

            d1, d2 = p.daughter1(), p.daughter2()
            has_same_flavour_daughter = False

            if d1 > 0 and d2 >= d1:
                for di in range(d1, d2 + 1):
                    if 0 <= di < pythia.event.size():
                        dau = pythia.event[di]
                        if abs(dau.id()) == pid_abs and abs(dau.status()) in RELEVANT_STATUS_ABS:
                            has_same_flavour_daughter = True
                            break

            if not has_same_flavour_daughter:
                partons.append((int(p.id()), float(p.eta()), float(p.phi()), float(p.pT())))

        return partons

    def _event_hardness_proxy_pt(self, partons):
        """
        Proxy de dureza del evento usando partones relevantes.
        Retorna max pT entre partones de matching.
        NO es exactamente PhaseSpace:pTHatMin de Pythia-only.
        """
        if not partons:
            return 0.0
        max_pt = 0.0
        for item in partons:
            if len(item) >= 4:
                pt = float(item[3])
                if pt > max_pt:
                    max_pt = pt
        return max_pt

    def _compute_fractions(self, const_info):
        """
        const_info: lista de (PseudoJet_constituent, pid, is_charged)
        """
        E_total = sum(cj.e() for cj, _, _ in const_info)
        if E_total <= 0.0:
            E_total = 1e-9

        E_chf = 0.0  # hadrónico cargado
        E_nef = 0.0  # EM neutro (fotones)
        E_nhf = 0.0  # hadrónico neutro
        E_cef = 0.0  # EM cargado (electrones)

        n_ch = 0
        n_neu = 0
        has_b = False
        has_c = False
        n_sv = 0
        muon_pt = 0.0

        for cj, pid, is_charged in const_info:
            e = cj.e()
            apid = abs(pid)

            if is_charged:
                n_ch += 1
                if apid == 11:
                    E_cef += e
                elif apid == 13:
                    muon_pt = max(muon_pt, cj.pt())
                else:
                    E_chf += e
            else:
                n_neu += 1
                if apid == 22:
                    E_nef += e
                else:
                    E_nhf += e

            if pid in B_HADRON_IDS:
                has_b = True
            if pid in C_HADRON_IDS:
                has_c = True
            if apid in LONG_LIVED_IDS_ABS:
                n_sv += 1

        return {
            "nef": float(np.clip(E_nef / E_total, 0.0, 1.0)),
            "nhf": float(np.clip(E_nhf / E_total, 0.0, 1.0)),
            "cef": float(np.clip(E_cef / E_total, 0.0, 1.0)),
            "chf": float(np.clip(E_chf / E_total, 0.0, 1.0)),
            "ncharged": int(n_ch),
            "nneutral": int(n_neu),
            "n_const": int(len(const_info)),
            "has_b": bool(has_b),
            "has_c": bool(has_c),
            "n_sv": int(n_sv),
            "muon_pt": float(muon_pt),
        }

    def _match_flavour(self, jet_eta, jet_phi, partons, R):
        """
        Matching con prioridad física:
          b > c > t > light quarks > gluón
        en dos etapas:
          1) cono estricto max(0.2, 0.4*R)
          2) cono completo R

        Devuelve PDG ID CON SIGNO.
        """
        priority = {5: 4, 4: 3, 6: 2, 1: 1, 2: 1, 3: 1, 21: 0}

        def best_in_cone(dr_max):
            best_pid = 0
            best_pri = -1
            best_dr = float("inf")

            for item in partons:
                pid, p_eta, p_phi = item[0], item[1], item[2]
                dphi = abs(p_phi - jet_phi)
                if dphi > np.pi:
                    dphi = 2 * np.pi - dphi
                dr = np.sqrt((p_eta - jet_eta) ** 2 + dphi ** 2)
                pri = priority.get(abs(pid), 0)

                if dr < dr_max and (pri > best_pri or (pri == best_pri and dr < best_dr)):
                    best_pid = pid
                    best_pri = pri
                    best_dr = dr

            return int(best_pid)

        strict_cone = max(0.2, 0.4 * R)
        pid_match = best_in_cone(strict_cone)
        if pid_match == 0:
            pid_match = best_in_cone(R)

        return int(pid_match)

    def _apply_detector_smearing(self, pt_true, eta_true, phi_true, m_true):
        """
        Smearing paramétrico tipo CMS/ATLAS usando self.rng (reproducible).
        """
        a = 1.0
        b = 0.05
        sigma_pt = pt_true * np.sqrt((a / np.sqrt(max(pt_true, 1.0))) ** 2 + b ** 2)

        reco_pt = max(0.0, self.rng.normal(pt_true, sigma_pt))
        reco_eta = float(self.rng.normal(eta_true, 0.01))
        reco_phi = wrap_phi(self.rng.normal(phi_true, 0.01))
        reco_m = max(0.0, self.rng.normal(m_true, 0.05 * max(m_true, 0.1)))

        return float(reco_pt), reco_eta, reco_phi, float(reco_m)

    def _compute_btag(self, fracs, flavour):
        if fracs["has_b"] or abs(flavour) == 5:
            return float(np.clip(self.rng.normal(0.85, 0.10), 0.0, 1.0))
        if fracs["has_c"] or abs(flavour) == 4:
            return float(np.clip(self.rng.normal(0.25, 0.10), 0.0, 1.0))
        return float(np.clip(self.rng.exponential(0.05), 0.0, 0.30))

    def _compute_ctag(self, fracs, flavour):
        if fracs["has_c"] or abs(flavour) == 4:
            return float(np.clip(self.rng.normal(0.80, 0.12), 0.0, 1.0))
        if fracs["has_b"] or abs(flavour) == 5:
            return float(np.clip(self.rng.normal(0.15, 0.08), 0.0, 1.0))
        return float(np.clip(self.rng.exponential(0.04), 0.0, 0.25))

    def _print_sanity(self, dataset, key):
        """
        Chequeos rápidos de sanidad física.
        dataset debe tener shape [N, 24].
        """
        if dataset.shape[0] == 0:
            print(f"    Sanity {key}: dataset vacío (0 jets).")
            return

        frac_sum = dataset[:, 11] + dataset[:, 12] + dataset[:, 13] + dataset[:, 14]
        bad_frac = np.mean((frac_sum < 0.5) | (frac_sum > 1.5))
        ratio = np.median(dataset[:, 6] / np.clip(dataset[:, 0], 1e-6, None))

        if dataset.shape[0] < 2 or np.std(dataset[:, 0]) == 0 or np.std(dataset[:, 6]) == 0:
            corr = np.nan
        else:
            corr = np.corrcoef(dataset[:, 0], dataset[:, 6])[0, 1]

        print(f"    Sanity {key}:")
        print(f"      pT corr gen/reco     : {corr:.4f}  (esperado > 0.95, si hay suficientes jets)")
        print(f"      pT ratio mediana     : {ratio:.3f}  (esperado ~1.0)")
        print(f"      Jets con fracs raras : {100 * bad_frac:.1f}%  (esperado bajo)")
        print(f"      Fracs EM+HAD mediana : {np.median(frac_sum):.3f}  (esperado ~1.0)")

    # --------------------------
    # Figuras
    # --------------------------
    def _plot_event_jets_eta_phi_from_arrays(self, jets_eta_phi_pt, cfg_key, source_event_idx, accepted_event_idx, out_dir):
        """
        jets_eta_phi_pt: lista de tuplas (eta, phi, pt)
        """
        if not jets_eta_phi_pt:
            return

        etas = np.array([x[0] for x in jets_eta_phi_pt], dtype=float)
        phis = np.array([x[1] for x in jets_eta_phi_pt], dtype=float)
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
        plt.title(f"{cfg_key} | srcEv={source_event_idx} | accEv={accepted_event_idx} | color=pT")
        plt.grid(True, alpha=0.25)
        cbar = plt.colorbar(sc)
        cbar.set_label(r"$p_T$ [GeV]")
        plt.tight_layout()

        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"srcEv_{source_event_idx:06d}_accEv_{accepted_event_idx:06d}_eta_phi.png")
        plt.savefig(out_path, dpi=140)
        plt.close()

    def _plot_global_dataset_figures(self, dataset, cfg_key, out_dir):
        """
        dataset shape [N,24]
        Usa columnas:
          0 pt_gen, 1 eta_gen, 2 phi_gen
        """
        if dataset.shape[0] == 0:
            return

        os.makedirs(out_dir, exist_ok=True)

        pt = dataset[:, 0]
        eta = dataset[:, 1]
        phi = dataset[:, 2]

        # Hist pT
        plt.figure(figsize=(6, 4))
        plt.hist(pt, bins=60, alpha=0.85)
        plt.xlabel(r"$p_T$ [GeV]")
        plt.ylabel("Jets")
        plt.title(f"{cfg_key} | pT")
        plt.grid(True, alpha=0.25)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "hist_pt.png"), dpi=140)
        plt.close()

        # Hist eta
        plt.figure(figsize=(6, 4))
        plt.hist(eta, bins=60, alpha=0.85)
        plt.xlabel(r"$\eta$")
        plt.ylabel("Jets")
        plt.title(f"{cfg_key} | eta")
        plt.grid(True, alpha=0.25)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "hist_eta.png"), dpi=140)
        plt.close()

        # Hist phi
        plt.figure(figsize=(6, 4))
        plt.hist(phi, bins=60, alpha=0.85)
        plt.xlabel(r"$\phi$")
        plt.ylabel("Jets")
        plt.title(f"{cfg_key} | phi")
        plt.grid(True, alpha=0.25)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "hist_phi.png"), dpi=140)
        plt.close()

        # Scatter eta-phi con color pT (sampleado si hay demasiados puntos)
        if dataset.shape[0] > self.max_scatter_points_global:
            idx = self.rng.choice(dataset.shape[0], size=self.max_scatter_points_global, replace=False)
            eta_s, phi_s, pt_s = eta[idx], phi[idx], pt[idx]
        else:
            eta_s, phi_s, pt_s = eta, phi, pt

        plt.figure(figsize=(7, 5))
        sc = plt.scatter(eta_s, phi_s, c=pt_s, s=8, alpha=0.65)
        plt.xlabel(r"$\eta$")
        plt.ylabel(r"$\phi$")
        plt.title(f"{cfg_key} | eta vs phi (color = pT)")
        plt.grid(True, alpha=0.25)
        cbar = plt.colorbar(sc)
        cbar.set_label(r"$p_T$ [GeV]")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "scatter_eta_phi_ptcolor.png"), dpi=140)
        plt.close()

    # --------------------------
    # Diagramas de Feynman
    # --------------------------
    def _convert_ps_eps_to_pdf_jpg(self, src_path, out_dir):
        """
        Convierte .ps/.eps a .pdf y .jpg si hay herramientas disponibles.
        Requiere:
          - ps2pdf (Ghostscript)
          - magick (ImageMagick) o convert
        """
        ext = os.path.splitext(src_path)[1].lower()
        if ext not in {".ps", ".eps"}:
            return []

        created = []
        base_name = os.path.splitext(os.path.basename(src_path))[0]

        ps2pdf_bin = shutil.which("ps2pdf")
        magick_bin = shutil.which("magick")
        convert_bin = shutil.which("convert")  # fallback

        # 1) PS/EPS -> PDF
        pdf_path = os.path.join(out_dir, f"{base_name}.pdf")
        if ps2pdf_bin:
            try:
                subprocess.run([ps2pdf_bin, src_path, pdf_path], check=True)
                if os.path.exists(pdf_path):
                    created.append(pdf_path)
            except Exception as e:
                print(f"  [diag] Error convirtiendo a PDF ({src_path}): {e}")
        else:
            print("  [diag] 'ps2pdf' no está instalado; no se pudo generar PDF.")

        # 2) PS/EPS -> JPG (vía ImageMagick)
        jpg_path = os.path.join(out_dir, f"{base_name}.jpg")
        if magick_bin or convert_bin:
            if magick_bin:
                cmd = [magick_bin, "-density", "300", src_path, "-quality", "95", jpg_path]
            else:
                cmd = [convert_bin, "-density", "300", src_path, "-quality", "95", jpg_path]

            try:
                subprocess.run(cmd, check=True)
                if os.path.exists(jpg_path):
                    created.append(jpg_path)
            except Exception as e:
                print(f"  [diag] Error convirtiendo a JPG ({src_path}): {e}")
                print("  [diag] Si ves 'not authorized', hay que habilitar PS/EPS en policy.xml de ImageMagick.")
        else:
            print("  [diag] 'magick'/'convert' no está instalado; no se pudo generar JPG.")

        return created

    def _collect_feynman_diagrams(self, run_dir):
        """
        Busca diagramas de MG5 y los copia a run_dir/feynman_diagrams.
        Además convierte .ps/.eps a .pdf/.jpg si es posible.
        """
        if not self.process_name:
            return

        proc_candidates = [
            os.path.join(self.work_dir, self.process_name),
            os.path.join(self.work_dir, os.path.basename(self.process_name)),
        ]
        proc_dir = None
        for c in proc_candidates:
            if os.path.isdir(c):
                proc_dir = c
                break

        if proc_dir is None:
            print("  [diag] No se encontró carpeta del proceso MG5 para recopilar diagramas.")
            return

        out_dir = os.path.join(run_dir, "feynman_diagrams")
        os.makedirs(out_dir, exist_ok=True)

        patterns = [
            os.path.join(proc_dir, "SubProcesses", "**", "*.pdf"),
            os.path.join(proc_dir, "SubProcesses", "**", "*.png"),
            os.path.join(proc_dir, "SubProcesses", "**", "*.jpg"),
            os.path.join(proc_dir, "SubProcesses", "**", "*.jpeg"),
            os.path.join(proc_dir, "SubProcesses", "**", "*.eps"),
            os.path.join(proc_dir, "SubProcesses", "**", "*.ps"),
            os.path.join(proc_dir, "HTML", "**", "*.pdf"),
            os.path.join(proc_dir, "HTML", "**", "*.png"),
            os.path.join(proc_dir, "HTML", "**", "*.jpg"),
            os.path.join(proc_dir, "HTML", "**", "*.jpeg"),
            os.path.join(proc_dir, "HTML", "**", "*.eps"),
            os.path.join(proc_dir, "HTML", "**", "*.ps"),
        ]

        found = []
        for pat in patterns:
            found.extend(glob.glob(pat, recursive=True))
        found = sorted(set(found))

        if not found:
            print("  [diag] No se encontraron diagramas en formatos comunes.")
            return

        copied = 0
        converted = 0

        for src in found:
            name = os.path.basename(src)
            dst = os.path.join(out_dir, name)

            if os.path.exists(dst):
                base, ext = os.path.splitext(name)
                k = 1
                while True:
                    dst_try = os.path.join(out_dir, f"{base}_{k}{ext}")
                    if not os.path.exists(dst_try):
                        dst = dst_try
                        break
                    k += 1

            try:
                shutil.copy2(src, dst)
                copied += 1

                # Si es .ps o .eps, intentar convertir a PDF/JPG
                made = self._convert_ps_eps_to_pdf_jpg(dst, out_dir)
                converted += len(made)

            except Exception as e:
                print(f"  [diag] No pude copiar {src}: {e}")

        print(f"  [diag] Diagramas copiados a: {out_dir} ({copied} archivos, {converted} conversiones)")

    # --------------------------
    # Procesamiento principal
    # --------------------------
    def process_with_pythia_fastjet(self):
        print("\n[3/4] PYTHIA + FASTJET")
        print("-" * 50)

        self._decompress_lhe_if_needed()
        print("  Inicializando Pythia8...")
        pythia = self._init_pythia()

        print("  Leyendo eventos de Pythia...")
        stored_events = []
        # Cada entrada:
        # {
        #   "source_event_idx": int,
        #   "accepted_event_idx": int,
        #   "particles": [(PseudoJet, pid, is_charged), ...],
        #   "particle_map": {user_index: (pid, is_charged)},
        #   "partons": [(pid, eta, phi, pt), ...],
        #   "hard_proxy_pt": float
        # }

        accepted_counter = 0
        for i_ev in range(self.n_events):
            if not pythia.next():
                print(f"  ⚠ Pythia terminó en intento {i_ev}")
                break

            particles = []
            particle_map = {}

            for i in range(pythia.event.size()):
                p = pythia.event[i]
                if not p.isFinal():
                    continue

                # Excluir neutrinos (jets visibles/reco-like)
                if abs(p.id()) in NEUTRINO_IDS_ABS:
                    continue

                px, py, pz, e = p.px(), p.py(), p.pz(), p.e()
                if (not np.isfinite(px) or not np.isfinite(py) or
                    not np.isfinite(pz) or not np.isfinite(e)):
                    continue
                if e <= 0.0:
                    continue

                pj = fj.PseudoJet(px, py, pz, e)
                pj.set_user_index(i)

                pid = int(p.id())
                is_charged = bool(p.isCharged())

                particles.append((pj, pid, is_charged))
                particle_map[i] = (pid, is_charged)

            partons = self._extract_partons_for_matching(pythia)
            hard_proxy_pt = self._event_hardness_proxy_pt(partons)

            # Filtro tipo "pTHat" proxy (solo si está activado)
            if self.min_hard_parton_pt > 0.0 and hard_proxy_pt < self.min_hard_parton_pt:
                continue

            stored_events.append({
                "source_event_idx": int(i_ev),
                "accepted_event_idx": int(accepted_counter),
                "particles": particles,
                "particle_map": particle_map,
                "partons": partons,
                "hard_proxy_pt": float(hard_proxy_pt),
            })
            accepted_counter += 1

            if (i_ev + 1) % 200 == 0:
                print(f"    {i_ev + 1}/{self.n_events} intentos leídos...  aceptados: {accepted_counter}")

        n_events_real = len(stored_events)
        print(f"  ✓ {n_events_real} eventos aceptados y almacenados")
        if self.min_hard_parton_pt > 0:
            print(f"    (Con hardness proxy >= {self.min_hard_parton_pt} GeV)")

        all_datasets = {}
        # all_datasets[key] = {
        #   "data": np.ndarray,
        #   "algo": str,
        #   "algo_code": int,
        #   "R": float,
        #   "event_figures": [...]
        # }

        for cfg in self.jet_configs:
            algo_name = cfg["algo_name"]
            algo_func = cfg["algo_func"]
            algo_code = int(cfg["algo_code"])
            R = float(cfg["R"])

            key = f"{algo_name}_R{R:g}"
            print(f"\n  Procesando {key} (pT_min={self.jet_pt_min} GeV)...")

            jet_def = fj.JetDefinition(algo_func, R)
            dataset_rows = []
            event_figures = []

            for event_data in stored_events:
                particles = event_data["particles"]
                particle_map = event_data["particle_map"]
                partons = event_data["partons"]

                pseudojets = [pj for pj, _, _ in particles]
                if not pseudojets:
                    continue

                cs = fj.ClusterSequence(pseudojets, jet_def)
                try:
                    jets_all = fj.sorted_by_pt(cs.inclusive_jets())
                except Exception:
                    jets_all = sorted(cs.inclusive_jets(), key=lambda j: -j.pt())

                jets = [j for j in jets_all if j.pt() >= self.jet_pt_min]
                if not jets:
                    continue

                if self.save_jet_figures and len(event_figures) < self.max_event_figures_per_cfg:
                    jets_eta_phi_pt = [(float(j.eta()), float(j.phi()), float(j.pt())) for j in jets]
                    event_figures.append({
                        "source_event_idx": int(event_data["source_event_idx"]),
                        "accepted_event_idx": int(event_data["accepted_event_idx"]),
                        "jets_eta_phi_pt": jets_eta_phi_pt,
                    })

                for jet in jets:
                    const_info = []
                    for cj in jet.constituents():
                        idx = cj.user_index()
                        info = particle_map.get(idx)
                        if info is None:
                            continue
                        pid, is_charged = info
                        const_info.append((cj, pid, is_charged))

                    if not const_info:
                        continue

                    fracs = self._compute_fractions(const_info)
                    flavour = self._match_flavour(jet.eta(), jet.phi(), partons, R)
                    rPt, rEta, rPhi, rM = self._apply_detector_smearing(
                        jet.pt(), jet.eta(), jet.phi(), jet.m()
                    )

                    jet_id = jet_quality_id(fracs)
                    qgl = quark_gluon_likelihood(fracs)

                    btag = self._compute_btag(fracs, flavour)
                    ctag = self._compute_ctag(fracs, flavour)

                    feat = np.array([
                        jet.pt(),                   # 0  pt_gen
                        jet.eta(),                  # 1  eta_gen
                        jet.phi(),                  # 2  phi_gen
                        jet.m(),                    # 3  m_gen
                        float(flavour),             # 4  flavour (PDG con signo)
                        btag,                       # 5
                        rPt,                        # 6  recoPt
                        rPhi,                       # 7  recoPhi
                        rEta,                       # 8  recoEta
                        fracs["muon_pt"],           # 9
                        float(fracs["n_const"]),    # 10
                        fracs["nef"],               # 11
                        fracs["nhf"],               # 12
                        fracs["cef"],               # 13
                        fracs["chf"],               # 14
                        qgl,                        # 15
                        float(jet_id),              # 16
                        float(fracs["ncharged"]),   # 17
                        float(fracs["nneutral"]),   # 18
                        ctag,                       # 19
                        float(fracs["n_sv"]),       # 20
                        rM,                         # 21 recoMass
                        float(R),                   # 22 jetR
                        float(algo_code),           # 23 algoCode
                    ], dtype=np.float32)

                    dataset_rows.append(feat)

            if dataset_rows:
                dataset = np.asarray(dataset_rows, dtype=np.float32)
            else:
                dataset = np.empty((0, N_FEATURES), dtype=np.float32)

            all_datasets[key] = {
                "data": dataset,
                "algo": algo_name,
                "algo_code": algo_code,
                "R": R,
                "event_figures": event_figures,
            }

            print(f"    -> {dataset.shape[0]} jets ({key})")
            self._print_sanity(dataset, key)

        try:
            pythia.stat()
        except Exception:
            pass

        return all_datasets, n_events_real

    # --------------------------
    # Guardado
    # --------------------------
    def save_datasets(self, all_datasets, n_events):
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        saved_npy_paths = []

        run_dir = os.path.join(self.work_dir, f"run_{self.process_name}_{ts}")
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

            base = os.path.join(cfg_dir, f"jets_{self.process_name}_{cfg_key}")

            # .npy principal
            npy_path = f"{base}.npy"
            np.save(npy_path, data)
            print(f"  ✓ {cfg_key}/jets_...npy  ({data.shape[0]} jets)")
            saved_npy_paths.append(npy_path)

            # metadata
            meta = {
                "timestamp": ts,
                "process": self.process_name,
                "n_events_accepted": int(n_events),
                "n_jets": int(data.shape[0]),
                "algorithm": algo,
                "algorithm_code": algo_code,
                "algorithm_code_map": {str(k): v for k, v in ALGO_CODE_TO_NAME.items()},
                "R": float(R),
                "jet_pt_min": float(self.jet_pt_min),
                "min_hard_parton_pt_proxy": float(self.min_hard_parton_pt),
                "n_features": int(N_FEATURES),
                "features": {str(i): name for i, name in enumerate(FEATURE_NAMES)},
                "feature_notes": {
                    "flavour": "PDG ID CON SIGNO (antipartículas negativas) como float32",
                    "reco": "Smearing parametrico sobre variables GEN",
                    "btag_ctag": "Proxies usando hadrones B/C y flavour matching",
                    "fractions": "NEF/NHF/CEF/CHF desde constituyentes finales de Pythia8",
                    "jetR": "Columna 22 con el valor de R usado para ese jet",
                    "algoCode": "Columna 23 con código del algoritmo (1=antikt,2=kt,3=cambridge)",
                },
                "notes": {
                    "visible_jets": "Neutrinos excluidos del clustering",
                    "flavour_matching": "Matching jet-parton con status abs in {23,33,43,51,52,53,59,62} y cono adaptativo",
                    "hardness_proxy": "NO equivale exactamente a PhaseSpace:pTHatMin de Pythia-only; es max pT de partón relevante",
                    "flowSim_hint": "Si FlowSim quiere enteros estrictos para flavour/algoCode, castear columnas 4 y 23 a int al cargar",
                },
            }

            with open(f"{base}_metadata.json", "w") as f:
                json.dump(meta, f, indent=2)

            # preview txt
            with open(f"{base}_preview.txt", "w") as f:
                f.write(f"# {cfg_key} | {data.shape[0]} jets | {n_events} eventos aceptados\n")
                f.write(f"# jet_pt_min = {self.jet_pt_min} GeV\n")
                f.write(f"# min_hard_parton_pt_proxy = {self.min_hard_parton_pt} GeV (0=off)\n")
                f.write(f"# algoCode map: 1=antikt, 2=kt, 3=cambridge\n")
                f.write("# " + "  ".join(f"{i}:{n}" for i, n in enumerate(FEATURE_NAMES)) + "\n")
                if data.shape[0] > 0:
                    np.savetxt(f, data[:10], fmt="%10.4f")
                else:
                    f.write("# Dataset vacío\n")

            # Figuras globales + por evento
            if self.save_jet_figures:
                fig_dir = os.path.join(cfg_dir, "figures")
                self._plot_global_dataset_figures(data, cfg_key, fig_dir)

                ev_fig_dir = os.path.join(cfg_dir, "event_figures")
                for item in event_figures:
                    self._plot_event_jets_eta_phi_from_arrays(
                        jets_eta_phi_pt=item["jets_eta_phi_pt"],
                        cfg_key=cfg_key,
                        source_event_idx=item["source_event_idx"],
                        accepted_event_idx=item["accepted_event_idx"],
                        out_dir=ev_fig_dir
                    )

        # README general
        readme_path = os.path.join(run_dir, "README.txt")
        with open(readme_path, "w") as f:
            f.write(f"Run: {self.process_name}\n")
            f.write(f"Fecha: {ts}\n")
            f.write(f"Eventos aceptados: {n_events}\n")
            f.write(f"Jet pT mínimo: {self.jet_pt_min} GeV\n")
            f.write(f"Hardness proxy (max pT partón): {self.min_hard_parton_pt} GeV (0=off)\n\n")

            f.write("Configuraciones generadas:\n")
            for cfg_key, payload in all_datasets.items():
                f.write(
                    f"  {cfg_key:25s} -> {payload['data'].shape[0]} jets"
                    f"   (algo={payload['algo']}, code={payload['algo_code']}, R={payload['R']})\n"
                )

            f.write("\nEstructura:\n")
            f.write(f"  run_{self.process_name}_{ts}/\n")
            for cfg_key in all_datasets:
                f.write(f"    {cfg_key}/\n")
                f.write("      jets_...npy\n")
                f.write("      jets_..._metadata.json\n")
                f.write("      jets_..._preview.txt\n")
                if self.save_jet_figures:
                    f.write("      figures/\n")
                    f.write("        hist_pt.png\n")
                    f.write("        hist_eta.png\n")
                    f.write("        hist_phi.png\n")
                    f.write("        scatter_eta_phi_ptcolor.png\n")
                    f.write("      event_figures/\n")
                    f.write("        srcEv_..._accEv_..._eta_phi.png\n")
            if self.save_feynman_diagrams:
                f.write("    feynman_diagrams/\n")
                f.write("      (copias y conversiones de diagramas encontrados en MG5)\n")

            f.write("\nColumnas del .npy:\n")
            for i, name in enumerate(FEATURE_NAMES):
                f.write(f"  [{i:2d}] {name}\n")

            f.write("\nMapa de algoritmo (algoCode):\n")
            for code, name in sorted(ALGO_CODE_TO_NAME.items()):
                f.write(f"  {code} -> {name}\n")

            f.write("\nNota importante para FlowSim:\n")
            f.write("  La columna [4] 'flavour' guarda el PDG ID con signo (float32).\n")
            f.write("  La columna [23] 'algoCode' también está en float32.\n")
            f.write("  Si tu pipeline espera enteros, convierte esas columnas a int al cargar.\n")

        # Diagramas de Feynman
        if self.save_feynman_diagrams:
            self._collect_feynman_diagrams(run_dir)

        print(f"\n  README -> {readme_path}")
        return run_dir, saved_npy_paths

    # --------------------------
    # Diagnóstico opcional
    # --------------------------
    def diagnose_parton_status(self, n_events_diag=5):
        """
        Útil si la columna flavour sale muy llena de ceros.
        """
        if not self.lhe_file:
            raise ValueError("Primero define self.lhe_file (o corre el workflow hasta cargar el LHE).")

        print("\nDIAGNÓSTICO DE STATUS DE PARTONES")
        print("-" * 50)

        self._decompress_lhe_if_needed()

        pythia = pythia8.Pythia()
        pythia.readString("Beams:frameType = 4")
        pythia.readString(f"Beams:LHEF = {self.lhe_file}")
        pythia.readString("HadronLevel:Hadronize = on")
        pythia.readString("Print:quiet = on")
        if not pythia.init():
            raise RuntimeError("No se pudo inicializar Pythia para diagnóstico.")

        status_counts = {}

        for _ in range(n_events_diag):
            if not pythia.next():
                break
            for i in range(pythia.event.size()):
                p = pythia.event[i]
                if abs(p.id()) in QUARK_GLUON_IDS_ABS:
                    s = int(p.status())
                    status_counts[s] = status_counts.get(s, 0) + 1

        print("  Status encontrados (quarks/gluón):")
        for s, c in sorted(status_counts.items()):
            bar = "█" * min(30, c)
            print(f"    status {s:+4d}: {c:5d}  {bar}")

        print("\n  Recuerda: el matching usa abs(status) en:")
        print(f"    {sorted(RELEVANT_STATUS_ABS)}")

    # --------------------------
    # Flujo principal
    # --------------------------
    def run(self):
        self.print_header()
        self.run_madgraph_interactive()

        if not self.configure_analysis():
            print("Análisis cancelado.")
            return

        all_datasets, n_ev = self.process_with_pythia_fastjet()
        run_dir, saved = self.save_datasets(all_datasets, n_ev)

        print("\n" + "=" * 80)
        print("  WORKFLOW COMPLETADO")
        print("=" * 80)
        print(f"  Proceso   : {self.process_name}")
        print(f"  Eventos   : {n_ev} (aceptados)")
        print(f"  pT min    : {self.jet_pt_min} GeV")
        print(f"  Hard proxy: {self.min_hard_parton_pt} GeV (0=off)")
        print(f"  Carpeta   : {run_dir}")
        print()

        for cfg_key, payload in all_datasets.items():
            arr = payload["data"]
            print(f"  {cfg_key:25s} -> {arr.shape[0]:6d} jets")

        print("\n  Archivos .npy:")
        for path in saved:
            print(f"    {path}")


if __name__ == "__main__":
    workflow = HEPWorkflow()
    try:
        workflow.run()
    except KeyboardInterrupt:
        print("\n\nWorkflow interrumpido por el usuario.")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()