"""Constants for the HEP workflow.

This module defines feature names, algorithm code maps, and particle ID sets.
It has no side effects and does not execute anything on import.
"""


FEATURE_NAMES = [
    "pt_gen", "eta_gen", "phi_gen", "m_gen", "flavour",
    "btag", "recoPt", "recoPhi", "recoEta", "muon_pT",
    "recoNConst", "nef", "nhf", "cef", "chf",
    "qgl", "jetId", "ncharged", "nneutral", "ctag",
    "nSV", "recoMass", "jetR", "algoCode"
]

N_FEATURES = len(FEATURE_NAMES)

ALGO_NAME_TO_CODE = {
    "antikt": 1,
    "kt": 2,
    "cambridge": 3,
}

ALGO_CODE_TO_NAME = {v: k for k, v in ALGO_NAME_TO_CODE.items()}

B_HADRON_IDS = {
    511, 521, 531, 541, 5122, 5132, 5232, 5332,
    -511, -521, -531, -541, -5122, -5132, -5232, -5332
}

C_HADRON_IDS = {
    411, 421, 431, 4122, 4132, 4232, 4332,
    -411, -421, -431, -4122, -4132, -4232, -4332
}

LONG_LIVED_IDS_ABS = {
    310, 130, 3122, 3112, 3222, 3312, 3334, 421, 411
}

QUARK_GLUON_IDS_ABS = {1, 2, 3, 4, 5, 6, 21}

RELEVANT_STATUS_ABS = {23, 33, 43, 51, 52, 53, 59, 62}

NEUTRINO_IDS_ABS = {12, 14, 16}


