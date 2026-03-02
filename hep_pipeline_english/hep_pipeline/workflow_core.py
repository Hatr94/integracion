"""Core workflow logic.

This is the main pipeline that runs MadGraph, then Pythia8, then FastJet.
The file is intentionally split from constants, helpers, plotting, and IO.
"""

import os
import glob
import gzip
import json
import shutil
import subprocess
from datetime import datetime

import numpy as np

import pythia8
import fastjet as fj

from .config_constants import (
    FEATURE_NAMES, N_FEATURES,
    ALGO_NAME_TO_CODE, ALGO_CODE_TO_NAME,
    B_HADRON_IDS, C_HADRON_IDS,
    LONG_LIVED_IDS_ABS, QUARK_GLUON_IDS_ABS,
    RELEVANT_STATUS_ABS, NEUTRINO_IDS_ABS,
)

from .utils_physics import wrap_phi, jet_quality_id, quark_gluon_likelihood

from .plotting import plot_event_jets_eta_phi_from_arrays, plot_global_dataset_figures
from .io_outputs import save_datasets
from .mg5_diagrams import collect_feynman_diagrams, convert_ps_eps_to_pdf_jpg



class HEPWorkflow:

    def __init__(self):

        self.mg5_path = os.environ.get(
            "MG5_PATH",
            os.path.expanduser("~/HEP/MG5_aMC_v3_7_0/bin/mg5_aMC")
        )

        self.work_dir = os.getcwd()

        self.lhe_file = None

        self.output_dir = None

        self.process_name = None

        self.jet_configs = [
            {
                "R": 0.4,
                "algo_name": "antikt",
                "algo_func": fj.antikt_algorithm,
                "algo_code": ALGO_NAME_TO_CODE["antikt"]
            },
            {
                "R": 0.8,
                "algo_name": "kt",
                "algo_func": fj.kt_algorithm,
                "algo_code": ALGO_NAME_TO_CODE["kt"]
            },
            {
                "R": 1.0,
                "algo_name": "cambridge",
                "algo_func": fj.cambridge_algorithm,
                "algo_code": ALGO_NAME_TO_CODE["cambridge"]
            },
        ]

        self.n_events = 1000

        self.jet_pt_min = 15.0

        self.min_hard_parton_pt = 0.0

        self.save_feynman_diagrams = True
        self.save_jet_figures = True
        self.max_event_figures_per_cfg = 30
        self.max_scatter_points_global = 20000

        self.rng = np.random.default_rng(42)
        self.pythia_seed = 42


    def print_header(self):
        print("\n" + "=" * 80)
        print("  WORKFLOW HEP: MadGraph to Pythia8 to FastJet (features fisicas)")
        print("=" * 80)
        print(f"  Working directory: {self.work_dir}")
        print("=" * 80 + "\n")

    def _parse_jet_configs_string(self, txt: str):
        """
        Formato expected:
          0.4:antikt,0.8:kt,1.0:cambridge

        alid algorithms:
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
                raise ValueError(f"Missing ':' in '{item}'. Use format R:algorithm")

            r_str, a_str = item.split(":", 1)

            R = float(r_str.strip())
            if R <= 0:
                raise ValueError(f"R invalid ({R}). Must be > 0.")

            algo_key = a_str.strip().lower()

            if algo_key not in algo_map:
                raise ValueError(
                    f"Invalid algorithm '{algo_key}'. Use antikt, kt, or cambridge."
                )

            algo_name = "cambridge" if algo_key in {"ca", "cam"} else algo_key

            out.append({
                "R": float(R),
                "algo_name": algo_name,
                "algo_func": algo_map[algo_key],
                "algo_code": ALGO_NAME_TO_CODE[algo_name],
            })

        if not out:
            raise ValueError("Could not parse any jet configuration.")

        return out

    def run_madgraph_interactive(self):
        print("[1/4] MADGRAPH INTERACTIVE")
        print("-" * 50)
        print("  - Generate your process normally")
        print("  - Use 'output NAME' for the output directory")
        print("  - When finished write 'quit'\n")

        if not os.path.isfile(self.mg5_path):
            raise FileNotFoundError(
                f"Could not find mg5_aMC at:\n  {self.mg5_path}\n"
                "Edita self.mg5_path o exporta MG5_PATH con la ruta correcta."
            )

        input("  Press ENTER to open MadGraph...")

        subprocess.run([self.mg5_path], check=False)

        raw_name = input("\nName of the output directory you used: ").strip().rstrip("/")

        if not raw_name:
            raise ValueError("You did not enter the MadGraph output directory name.")

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
            print(f"   Found LHE file: {self.lhe_file}")
            self.output_dir = os.path.dirname(self.lhe_file)
        else:
            self.lhe_file = input("  Full path to the .lhe or .lhe.gz file: ").strip()
            if not self.lhe_file:
                raise ValueError("No path was provided for the .lhe or .lhe.gz file.")
            if not os.path.exists(self.lhe_file):
                raise FileNotFoundError(f"Does not exist: {self.lhe_file}")
            self.output_dir = os.path.dirname(self.lhe_file)

    def configure_analysis(self) -> bool:
        print("\n[2/4] CONFIGURATION")
        print("-" * 50)

        n_in = input(f"  Events to attempt (Pythia attempts) (default {self.n_events}): ").strip()
        if n_in:
            self.n_events = int(n_in)
            if self.n_events <= 0:
                raise ValueError("Number of events must be > 0.")

        pt_in = input(f"  Minimum jet pT [GeV] (default {self.jet_pt_min}): ").strip()
        if pt_in:
            self.jet_pt_min = float(pt_in)
            if self.jet_pt_min < 0:
                raise ValueError("Minimum jet pT must be >= 0.")

        hard_in = input(
            f"  Event hardness proxy cut (max pT parton) [GeV] (default {self.min_hard_parton_pt}, 0=off): "
        ).strip()
        if hard_in:
            self.min_hard_parton_pt = float(hard_in)
            if self.min_hard_parton_pt < 0:
                raise ValueError("The event hardness proxy cut must be >= 0.")

        print("\n  Jet configuration by R (editable)")
        print("    Format: R:algorithm,R:algorithm,...")
        print("    Valid algorithms: antikt, kt, cambridge")
        print("    Example: 0.4:antikt,0.8:kt,1.0:cambridge")

        default_cfg_str = ",".join(f'{cfg["R"]}:{cfg["algo_name"]}' for cfg in self.jet_configs)

        cfg_in = input(f"  Configs jets (default {default_cfg_str}): ").strip()
        if cfg_in:
            self.jet_configs = self._parse_jet_configs_string(cfg_in)

        figs_in = input("  Saver figures de jets (globales + por evento)? (s/n, default s): ").strip().lower()
        if figs_in in {"n", "no"}:
            self.save_jet_figures = False

        diag_in = input("  Copy/convert Feynman diagrams from MG5 if they exist? (s/n, default s): ").strip().lower()
        if diag_in in {"n", "no"}:
            self.save_feynman_diagrams = False

        if self.save_jet_figures:
            nevfig_in = input(
                f"  Max. figures per event per configuration (default {self.max_event_figures_per_cfg}): "
            ).strip()
            if nevfig_in:
                self.max_event_figures_per_cfg = int(nevfig_in)
                if self.max_event_figures_per_cfg < 0:
                    raise ValueError("max_event_figures_per_cfg debe ser >= 0.")

        print("\n  summary:")
        print(f"    Events (attempts) : {self.n_events}")
        print(f"    pT min jets        : {self.jet_pt_min} GeV")
        print(f"    Hardness proxy     : {self.min_hard_parton_pt} GeV (0=off)")
        print(f"    Jet figures       : {self.save_jet_figures}")
        print(f"    MG5 Diagrams      : {self.save_feynman_diagrams}")
        print("    Jet configs        :")
        for cfg in self.jet_configs:
            print(f'      R={cfg["R"]}  to  {cfg["algo_name"]} (code={cfg["algo_code"]})')

        ans = input("\n  Continue? (s/n): ").strip().lower()

        return ans in {"s", "si", "si", "y", "yes"}


    def _decompress_lhe_if_needed(self):
        if self.lhe_file.endswith(".gz"):
            out_path = self.lhe_file[:-3]
            if os.path.exists(out_path):
                print(f"  LHE already decompressed, using: {out_path}")
            else:
                print(f"  Decompressing LHE to {out_path}")
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

        p.readString("Random:setSeed = on")
        p.readString(f"Random:seed = {self.pythia_seed}")

        p.readString("Print:quiet = on")

        if not p.init():
            raise RuntimeError("Pythia8 could not be initialized with the given LHE file.")

        return p

    def _extract_partons_for_matching(self, pythia):
        """
        Extract relevant partons (quarks and gluons) while avoiding duplicates along decay chains.

        Returns tuples: (pid, eta, phi, pt)
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
        Event hardness proxy using relevant partons.
        Returns max pT among partons used for matching.
        This is not exactly Pythia-only PhaseSpace:pTHatMin.
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

        E_chf = 0.0  # hadronico cargado
        E_nef = 0.0  # EM neutro (fotones)
        E_nhf = 0.0  # hadronico neutro
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
        Matching with physics motivated priority:
          b > c > t > light quarks > gluon
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
        Parametric CMS or ATLAS like smearing using self.rng (reproducible).
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
        Quick physics sanity checks.
        dataset debe tener shape [N, 24].
        """

        if dataset.shape[0] == 0:
            print(f"    Sanity {key}: dataset vacio (0 jets).")
            return

        frac_sum = dataset[:, 11] + dataset[:, 12] + dataset[:, 13] + dataset[:, 14]

        bad_frac = np.mean((frac_sum < 0.5) | (frac_sum > 1.5))

        ratio = np.median(dataset[:, 6] / np.clip(dataset[:, 0], 1e-6, None))

        if dataset.shape[0] < 2 or np.std(dataset[:, 0]) == 0 or np.std(dataset[:, 6]) == 0:
            corr = np.nan
        else:
            corr = np.corrcoef(dataset[:, 0], dataset[:, 6])[0, 1]

        print(f"    Sanity {key}:")
        print(f"      pT corr gen/reco     : {corr:.4f}  (expected > 0.95, if there are enough jets)")
        print(f"      median pT ratio     : {ratio:.3f}  (expected ~1.0)")
        print(f"      Jets con fracs raras : {100 * bad_frac:.1f}%  (expected bajo)")
        print(f"      Fracs EM+HAD mediana : {np.median(frac_sum):.3f}  (expected ~1.0)")


    def _plot_event_jets_eta_phi_from_arrays(self, jets_eta_phi_pt, cfg_key, source_event_idx, accepted_event_idx, out_dir):
        return plot_event_jets_eta_phi_from_arrays(jets_eta_phi_pt, cfg_key, source_event_idx, accepted_event_idx, out_dir)

    def _plot_global_dataset_figures(self, dataset, cfg_key, out_dir):
        return plot_global_dataset_figures(self, dataset, cfg_key, out_dir)

    def _convert_ps_eps_to_pdf_jpg(self, src_path, out_dir):
        return convert_ps_eps_to_pdf_jpg(self, src_path, out_dir)

    def _collect_feynman_diagrams(self, run_dir):
        return collect_feynman_diagrams(self, run_dir)

    def process_with_pythia_fastjet(self):
        print("\n[3/4] PYTHIA + FASTJET")
        print("-" * 50)

        self._decompress_lhe_if_needed()

        print("  Inicializando Pythia8...")
        pythia = self._init_pythia()

        print("  Leyendo eventos de Pythia...")
        stored_events = []


        accepted_counter = 0

        for i_ev in range(self.n_events):
            if not pythia.next():
                print(f"   Pythia termino en intento {i_ev}")
                break

            particles = []
            particle_map = {}

            for i in range(pythia.event.size()):
                p = pythia.event[i]

                if not p.isFinal():
                    continue

                if abs(p.id()) in NEUTRINO_IDS_ABS:
                    continue

                px, py, pz, e = p.px(), p.py(), p.pz(), p.e()

                if (
                    not np.isfinite(px) or not np.isfinite(py) or
                    not np.isfinite(pz) or not np.isfinite(e)
                ):
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
                print(f"    {i_ev + 1}/{self.n_events} Attempts read... accepted: {accepted_counter}")

        n_events_real = len(stored_events)
        print(f"   {n_events_real} accepted and stored events")
        if self.min_hard_parton_pt > 0:
            print(f"    (Con hardness proxy >= {self.min_hard_parton_pt} GeV)")

        all_datasets = {}


        for cfg in self.jet_configs:
            algo_name = cfg["algo_name"]
            algo_func = cfg["algo_func"]
            algo_code = int(cfg["algo_code"])
            R = float(cfg["R"])

            key = f"{algo_name}_R{R:g}"
            print(f"\n  Processing {key} (pT_min={self.jet_pt_min} GeV)...")

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
                        wrap_phi(jet.phi()),        # 2  phi_gen
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

            print(f"    to {dataset.shape[0]} jets ({key})")
            self._print_sanity(dataset, key)

        try:
            pythia.stat()
        except Exception:
            pass

        return all_datasets, n_events_real


    def save_datasets(self, all_datasets, n_events):
        return save_datasets(self, all_datasets, n_events)

    def diagnose_parton_status(self, n_events_diag=5):
        """
        Useful if the flavour column ends up filled with zeros.
        """

        if not self.lhe_file:
            raise ValueError("First set self.lhe_file (or run the workflow until the LHE is loaded).")

        print("\nDIAGNOSTICO DE STATUS DE PARTONES")
        print("-" * 50)

        self._decompress_lhe_if_needed()

        pythia = pythia8.Pythia()
        pythia.readString("Beams:frameType = 4")
        pythia.readString(f"Beams:LHEF = {self.lhe_file}")
        pythia.readString("HadronLevel:Hadronize = on")
        pythia.readString("Print:quiet = on")
        if not pythia.init():
            raise RuntimeError("Could not initialize Pythia for diagnostics.")

        status_counts = {}

        for _ in range(n_events_diag):
            if not pythia.next():
                break
            for i in range(pythia.event.size()):
                p = pythia.event[i]
                if abs(p.id()) in QUARK_GLUON_IDS_ABS:
                    s = int(p.status())
                    status_counts[s] = status_counts.get(s, 0) + 1

        print("  Status encontrados (quarks/gluon):")
        for s, c in sorted(status_counts.items()):
            bar = "" * min(30, c)
            print(f"    status {s:+4d}: {c:5d}  {bar}")

        print("\n  Reminder: matching uses abs(status) for:")
        print(f"    {sorted(RELEVANT_STATUS_ABS)}")


    def run(self):
        self.print_header()
        self.run_madgraph_interactive()

        if not self.configure_analysis():
            print("Analisis cancelado.")
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
            print(f"  {cfg_key:25s} to {arr.shape[0]:6d} jets")

        print("\n  .npy files:")
        for path in saved:
            print(f"    {path}")
