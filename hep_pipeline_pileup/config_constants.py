"""Constantes y tablas (PDG IDs, nombres de features, maps de algoritmos).

Este módulo no ejecuta nada: solo define constantes importables.
"""

# ==========================
# Constantes físicas / PDG IDs
# ==========================
# ↑ Sección con nombres de columnas y conjuntos PDG usados como proxies físicos.

FEATURE_NAMES = [
    # ↑ Lista de nombres de features en el orden exacto en que se guardan en el .npy.
    "pt_gen", "eta_gen", "phi_gen", "m_gen", "flavour",
    # ↑ Variables "GEN-like" y el flavour del jet (matching a partón, PDG con signo).
    "btag", "recoPt", "recoPhi", "recoEta", "muon_pT",
    # ↑ Proxies de detector / b-tag / variables auxiliares.
    "recoNConst", "nef", "nhf", "cef", "chf",
    # ↑ Número de constituyentes reconstruidos y fracciones de energía.
    "qgl", "jetId", "ncharged", "nneutral", "ctag",
    # ↑ QGL proxy, calidad de jet, multiplicidades cargadas/neutras y c-tag proxy.
    "nSV", "recoMass", "jetR", "algoCode",
    # ↑ Número de vértices secundarios proxy, masa reco, R usado y código de algoritmo.
    "mu", "pu_fraction",
    # ↑ [24] mu: NPU real del evento (condicionante FlowSim).
    # ↑ [25] pu_fraction: fracción de constituyentes del jet provenientes de PU.
]

N_FEATURES = len(FEATURE_NAMES)
# ↑ Número total de features (debe ser 24). Se calcula automáticamente para evitar errores.

# Mapeo numérico de algoritmo (para guardar en columna algoCode)
# ↑ Diccionario para convertir nombre de algoritmo a código numérico compacto.
ALGO_NAME_TO_CODE = {
    "antikt": 1,
    # ↑ anti-kT se guarda como 1.
    "kt": 2,
    # ↑ kT se guarda como 2.
    "cambridge": 3,
    # ↑ Cambridge/Aachen se guarda como 3.
}

ALGO_CODE_TO_NAME = {v: k for k, v in ALGO_NAME_TO_CODE.items()}
# ↑ Diccionario inverso (código -> nombre) para metadatos y lectura humana.

# Hadrones con quark b (proxy b-tag)
# ↑ Conjunto de PDG IDs de hadrones B para inferir presencia de contenido b dentro del jet.
B_HADRON_IDS = {
    511, 521, 531, 541, 5122, 5132, 5232, 5332,
    # ↑ Mesones/bariones B (partículas).
    -511, -521, -531, -541, -5122, -5132, -5232, -5332
    # ↑ Antipartículas correspondientes (signo negativo en PDG).
}

# Hadrones con quark c (proxy c-tag)
# ↑ Conjunto de PDG IDs de hadrones C para inferir contenido charm dentro del jet.
C_HADRON_IDS = {
    411, 421, 431, 4122, 4132, 4232, 4332,
    # ↑ Mesones/bariones con quark c.
    -411, -421, -431, -4122, -4132, -4232, -4332
    # ↑ Antipartículas charm.
}

# Partículas de vida larga (proxy de vértices secundarios)
# ↑ IDs absolutos típicos de partículas de vida relativamente larga para proxy de nSV.
LONG_LIVED_IDS_ABS = {
    310, 130, 3122, 3112, 3222, 3312, 3334, 421, 411
    # ↑ Kaones neutros, hipérones, hadrones charm, etc. (valores absolutos).
}

# Partones candidatos para flavour matching
# ↑ IDs absolutos de quarks (1-6) y gluón (21) para asignar flavour al jet.
QUARK_GLUON_IDS_ABS = {1, 2, 3, 4, 5, 6, 21}

# Status relevantes de Pythia para partones "finales" antes de hadronización / resonancias
# ↑ Se usan status (en valor absoluto) como filtro de partones físicamente útiles para matching.
RELEVANT_STATUS_ABS = {23, 33, 43, 51, 52, 53, 59, 62}

# Invisibles típicos (no entran en jets visibles)
# ↑ Neutrinos se excluyen del clustering porque no depositan energía visible en detector.
NEUTRINO_IDS_ABS = {12, 14, 16}


