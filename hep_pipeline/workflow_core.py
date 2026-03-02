"""Core del workflow: MadGraph -> Pythia8 -> FastJet.

Esta versión se refactorizó en varios módulos (constants/utils/plot/io/diagrams)
para que sea más fácil de leer y mantener.
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

# Funciones separadas (wrappers dentro de la clase)
from .plotting import plot_event_jets_eta_phi_from_arrays, plot_global_dataset_figures
from .io_outputs import save_datasets
from .mg5_diagrams import collect_feynman_diagrams, convert_ps_eps_to_pdf_jpg


# ==========================
# Workflow principal
# ==========================
# ↑ Clase principal que encapsula toda la lógica del flujo HEP.

class HEPWorkflow:
    # ↑ Define la clase del workflow.

    def __init__(self):
        # ↑ Constructor: inicializa rutas, configuración por defecto y RNG.

        # Puedes sobrescribir la ruta con variable de entorno MG5_PATH si quieres
        # ↑ Permite personalizar la ruta de MG5 sin tocar el código, usando una env var.
        self.mg5_path = os.environ.get(
            "MG5_PATH",
            # ↑ Primero intenta leer MG5_PATH del entorno.
            os.path.expanduser("~/HEP/MG5_aMC_v3_7_0/bin/mg5_aMC")
            # ↑ Si no existe, usa esta ruta por defecto (ajústala a tu instalación).
        )

        self.work_dir = os.getcwd()
        # ↑ Directorio de trabajo actual; se usa como raíz para buscar/guardar archivos.

        self.lhe_file = None
        # ↑ Ruta al archivo LHE (se definirá luego de correr MG5 o al ingresarlo manualmente).

        self.output_dir = None
        # ↑ Directorio donde está el LHE; se completará cuando se localice el archivo.

        self.process_name = None
        # ↑ Nombre del proceso/directorio de salida de MG5 (ej. ttbar_had_4j).

        # Configuración por defecto EDITABLE: (R -> algoritmo)
        # ↑ Lista de configuraciones de jets que se procesarán por defecto.
        self.jet_configs = [
            {
                "R": 0.4,
                # ↑ Radio del jet.
                "algo_name": "antikt",
                # ↑ Nombre del algoritmo (legible).
                "algo_func": fj.antikt_algorithm,
                # ↑ Constante/función de FastJet para anti-kT.
                "algo_code": ALGO_NAME_TO_CODE["antikt"]
                # ↑ Código numérico que se guardará en la columna 23.
            },
            {
                "R": 0.8,
                # ↑ Segundo jet radius.
                "algo_name": "kt",
                # ↑ Algoritmo kT.
                "algo_func": fj.kt_algorithm,
                # ↑ Implementación FastJet de kT.
                "algo_code": ALGO_NAME_TO_CODE["kt"]
                # ↑ Código 2.
            },
            {
                "R": 1.0,
                # ↑ Tercer radio.
                "algo_name": "cambridge",
                # ↑ Cambridge/Aachen.
                "algo_func": fj.cambridge_algorithm,
                # ↑ Implementación FastJet de Cambridge.
                "algo_code": ALGO_NAME_TO_CODE["cambridge"]
                # ↑ Código 3.
            },
        ]

        self.n_events = 1000
        # ↑ Número de intentos de eventos Pythia a procesar por defecto.

        self.jet_pt_min = 15.0
        # ↑ Corte mínimo de pT para jets reconstruidos (en GeV).

        # Proxy de "dureza" del evento (NO es exactamente pTHatMin de Pythia-only)
        # ↑ Se aclara que este filtro es un proxy calculado por partones relevantes.
        self.min_hard_parton_pt = 0.0
        # ↑ Si es 0, el filtro de dureza está desactivado.

        # Figuras y outputs extra
        # ↑ Flags y límites de outputs visuales.
        self.save_feynman_diagrams = True
        # ↑ Si True, intentará copiar/convertir diagramas de MG5.
        self.save_jet_figures = True
        # ↑ Si True, guarda histogramas globales y figuras por evento.
        self.max_event_figures_per_cfg = 30
        # ↑ Límite de figuras por evento por configuración para no explotar disco.
        self.max_scatter_points_global = 20000
        # ↑ Límite de puntos en scatter global (muestreo si hay demasiados jets).

        # RNG reproducible
        # ↑ Generador aleatorio reproducible para smearing y proxies.
        self.rng = np.random.default_rng(42)
        # ↑ Semilla fija para NumPy Generator.
        self.pythia_seed = 42
        # ↑ Semilla fija para Pythia (importante para reproducibilidad total).

    # --------------------------
    # UI
    # --------------------------
    # ↑ Métodos de interacción por consola con el usuario.

    def print_header(self):
        # ↑ Imprime encabezado bonito del workflow en terminal.
        print("\n" + "=" * 80)
        # ↑ Línea decorativa.
        print("  WORKFLOW HEP: MadGraph -> Pythia8 -> FastJet (features físicas)")
        # ↑ Título principal.
        print("=" * 80)
        # ↑ Línea decorativa.
        print(f"  Directorio de trabajo: {self.work_dir}")
        # ↑ Muestra el directorio actual para evitar confusiones de rutas.
        print("=" * 80 + "\n")
        # ↑ Cierra el encabezado con espacio visual.

    def _parse_jet_configs_string(self, txt: str):
        # ↑ Parsea una cadena del tipo "0.4:antikt,0.8:kt,1.0:cambridge".
        """
        Formato esperado:
          0.4:antikt,0.8:kt,1.0:cambridge

        Algoritmos válidos:
          antikt, kt, cambridge
        """
        # ↑ Docstring con formato de entrada esperado.

        algo_map = {
            "antikt": fj.antikt_algorithm,
            # ↑ Alias directo para anti-kT.
            "kt": fj.kt_algorithm,
            # ↑ Alias directo para kT.
            "cambridge": fj.cambridge_algorithm,
            # ↑ Nombre completo Cambridge/Aachen.
            "ca": fj.cambridge_algorithm,
            # ↑ Alias corto "ca".
            "cam": fj.cambridge_algorithm,
            # ↑ Alias corto "cam".
        }

        out = []
        # ↑ Aquí se irán acumulando las configuraciones parseadas.

        for item in txt.split(","):
            # ↑ Se separa por comas cada bloque "R:algoritmo".
            item = item.strip()
            # ↑ Se limpian espacios alrededor.

            if not item:
                # ↑ Si quedó vacío (por coma extra), se ignora.
                continue

            if ":" not in item:
                # ↑ Valida que exista el separador R:algoritmo.
                raise ValueError(f"Falta ':' en '{item}'. Usa formato R:algoritmo")

            r_str, a_str = item.split(":", 1)
            # ↑ Se separa en dos partes: radio y nombre del algoritmo.

            R = float(r_str.strip())
            # ↑ Convierte el radio a float.
            if R <= 0:
                # ↑ Validación física/técnica: el radio debe ser positivo.
                raise ValueError(f"R inválido ({R}). Debe ser > 0.")

            algo_key = a_str.strip().lower()
            # ↑ Normaliza el nombre del algoritmo a minúsculas.

            if algo_key not in algo_map:
                # ↑ Valida que el algoritmo esté soportado.
                raise ValueError(
                    f"Algoritmo inválido '{algo_key}'. Usa antikt, kt o cambridge."
                )

            algo_name = "cambridge" if algo_key in {"ca", "cam"} else algo_key
            # ↑ Si usaron alias ca/cam, se normaliza al nombre oficial "cambridge".

            out.append({
                "R": float(R),
                # ↑ Guarda el radio del jet.
                "algo_name": algo_name,
                # ↑ Guarda el nombre normalizado del algoritmo.
                "algo_func": algo_map[algo_key],
                # ↑ Guarda la referencia al algoritmo FastJet.
                "algo_code": ALGO_NAME_TO_CODE[algo_name],
                # ↑ Guarda el código numérico para el dataset.
            })

        if not out:
            # ↑ Si no se pudo parsear ninguna configuración válida, aborta con error.
            raise ValueError("No se pudo parsear ninguna configuración de jets.")

        return out
        # ↑ Devuelve la lista de configuraciones ya lista para usar.

    def run_madgraph_interactive(self):
        # ↑ Abre MG5 en modo interactivo y luego intenta localizar el archivo LHE generado.
        print("[1/4] MADGRAPH INTERACTIVO")
        # ↑ Título de paso 1.
        print("-" * 50)
        # ↑ Separador visual.
        print("  • Genera tu proceso normalmente")
        # ↑ Instrucción para el usuario.
        print("  • Usa 'output NOMBRE' para el directorio de salida")
        # ↑ Recordatorio de cómo nombrar la carpeta.
        print("  • Al terminar escribe 'quit'\n")
        # ↑ Recordatorio de salida de MG5.

        if not os.path.isfile(self.mg5_path):
            # ↑ Verifica que la ruta al ejecutable mg5_aMC realmente exista.
            raise FileNotFoundError(
                f"No encuentro mg5_aMC en:\n  {self.mg5_path}\n"
                "Edita self.mg5_path o exporta MG5_PATH con la ruta correcta."
            )

        input("  Presiona ENTER para abrir MadGraph...")
        # ↑ Pausa para que el usuario esté listo antes de abrir MG5.

        subprocess.run([self.mg5_path], check=False)
        # ↑ Lanza el ejecutable de MG5. check=False para no romper si MG5 sale con código no-cero.

        raw_name = input("\nNombre del directorio de salida que usaste: ").strip().rstrip("/")
        # ↑ Pide al usuario el nombre de la carpeta 'output' de MG5 y limpia espacios/slash final.

        if not raw_name:
            # ↑ Si no ingresó nada, se levanta error (es necesario para buscar el LHE).
            raise ValueError("No ingresaste el nombre del directorio de salida de MadGraph.")

        self.process_name = os.path.basename(raw_name)
        # ↑ Se extrae el nombre base por si ingresó una ruta completa.

        patterns = [
            # ↑ Lista de patrones de búsqueda del LHE (casos comunes y recursivos).
            os.path.join(self.work_dir, self.process_name, "Events", "run_01", "unweighted_events.lhe.gz"),
            # ↑ Caso típico comprimido.
            os.path.join(self.work_dir, self.process_name, "Events", "run_01", "unweighted_events.lhe"),
            # ↑ Caso típico sin comprimir.
            os.path.join(self.work_dir, "**", self.process_name, "**", "unweighted_events.lhe.gz"),
            # ↑ Búsqueda recursiva comprimida si el layout no es el esperado.
            os.path.join(self.work_dir, "**", self.process_name, "**", "unweighted_events.lhe"),
            # ↑ Búsqueda recursiva sin comprimir.
        ]

        for pat in patterns:
            # ↑ Recorre cada patrón hasta encontrar coincidencias.
            hits = glob.glob(pat, recursive=True)
            # ↑ Busca archivos que cumplan ese patrón.
            if hits:
                # ↑ Si encontró alguno...
                hits = sorted(hits, key=os.path.getmtime, reverse=True)
                # ↑ Ordena por fecha de modificación (más reciente primero).
                self.lhe_file = hits[0]
                # ↑ Selecciona el más reciente (probablemente el correcto).
                break
                # ↑ Sale del loop al encontrar el primer patrón útil.

        if self.lhe_file:
            # ↑ Si se encontró automáticamente el LHE...
            print(f"  ✓ Archivo LHE encontrado: {self.lhe_file}")
            # ↑ Lo muestra para confirmación visual.
            self.output_dir = os.path.dirname(self.lhe_file)
            # ↑ Guarda carpeta del LHE.
        else:
            # ↑ Si no se encontró por patrones, se pide la ruta manual.
            self.lhe_file = input("  Ruta completa al .lhe o .lhe.gz: ").strip()
            # ↑ El usuario puede pegar una ruta absoluta/manual.
            if not self.lhe_file:
                # ↑ Valida que no quede vacío.
                raise ValueError("No se proporcionó ruta al archivo .lhe/.lhe.gz.")
            if not os.path.exists(self.lhe_file):
                # ↑ Valida que exista en disco.
                raise FileNotFoundError(f"No existe: {self.lhe_file}")
            self.output_dir = os.path.dirname(self.lhe_file)
            # ↑ Guarda la carpeta del LHE.

    def configure_analysis(self) -> bool:
        # ↑ Pide al usuario parámetros de análisis (n eventos, pT min, configs de jet, etc.).
        print("\n[2/4] CONFIGURACIÓN")
        # ↑ Título del paso 2.
        print("-" * 50)
        # ↑ Separador visual.

        n_in = input(f"  Eventos a procesar (intentos Pythia) (default {self.n_events}): ").strip()
        # ↑ Pide cantidad de intentos/eventos Pythia; si vacío se usa el default.
        if n_in:
            # ↑ Solo convierte si el usuario escribió algo.
            self.n_events = int(n_in)
            # ↑ Convierte a entero.
            if self.n_events <= 0:
                # ↑ Valida que sea positivo.
                raise ValueError("El número de eventos debe ser > 0.")

        pt_in = input(f"  pT mínimo de jets [GeV] (default {self.jet_pt_min}): ").strip()
        # ↑ Pide el corte de pT de jets.
        if pt_in:
            # ↑ Solo actualiza si se escribió algo.
            self.jet_pt_min = float(pt_in)
            # ↑ Convierte a float.
            if self.jet_pt_min < 0:
                # ↑ Debe ser no negativo.
                raise ValueError("El pT mínimo debe ser >= 0.")

        hard_in = input(
            f"  Corte proxy de dureza del evento (max pT partón) [GeV] (default {self.min_hard_parton_pt}, 0=off): "
        ).strip()
        # ↑ Pide el corte del proxy de dureza (max pT partón relevante).
        if hard_in:
            # ↑ Solo actualiza si hay valor.
            self.min_hard_parton_pt = float(hard_in)
            # ↑ Convierte a float.
            if self.min_hard_parton_pt < 0:
                # ↑ Debe ser no negativo.
                raise ValueError("El corte proxy de dureza debe ser >= 0.")

        print("\n  Configuración de jets por R (editable)")
        # ↑ Encabezado del bloque de configuración de algoritmos/R.
        print("    Formato: R:algoritmo,R:algoritmo,...")
        # ↑ Explica el formato de entrada.
        print("    Algoritmos válidos: antikt, kt, cambridge")
        # ↑ Enumera algoritmos aceptados.
        print("    Ejemplo: 0.4:antikt,0.8:kt,1.0:cambridge")
        # ↑ Muestra ejemplo concreto.

        default_cfg_str = ",".join(f'{cfg["R"]}:{cfg["algo_name"]}' for cfg in self.jet_configs)
        # ↑ Construye la representación textual de las configs actuales para mostrarla como default.

        cfg_in = input(f"  Configs jets (default {default_cfg_str}): ").strip()
        # ↑ Pide al usuario la cadena de configuraciones.
        if cfg_in:
            # ↑ Si el usuario escribe algo, se parsea y reemplaza la configuración.
            self.jet_configs = self._parse_jet_configs_string(cfg_in)

        figs_in = input("  ¿Guardar figuras de jets (globales + por evento)? (s/n, default s): ").strip().lower()
        # ↑ Pregunta si se quieren guardar figuras.
        if figs_in in {"n", "no"}:
            # ↑ Si responde no, desactiva la generación de figuras.
            self.save_jet_figures = False

        diag_in = input("  ¿Copiar/convertir diagramas de Feynman de MG5 si existen? (s/n, default s): ").strip().lower()
        # ↑ Pregunta si se quieren recopilar diagramas de MG5.
        if diag_in in {"n", "no"}:
            # ↑ Si responde no, se desactiva ese paso.
            self.save_feynman_diagrams = False

        if self.save_jet_figures:
            # ↑ Solo pregunta por límite de figuras/evento si se van a guardar figuras.
            nevfig_in = input(
                f"  Máx. figuras por evento por configuración (default {self.max_event_figures_per_cfg}): "
            ).strip()
            # ↑ Pide el máximo de figures per cfg.
            if nevfig_in:
                # ↑ Si el usuario especifica valor...
                self.max_event_figures_per_cfg = int(nevfig_in)
                # ↑ Se actualiza el límite.
                if self.max_event_figures_per_cfg < 0:
                    # ↑ Debe ser no negativo.
                    raise ValueError("max_event_figures_per_cfg debe ser >= 0.")

        print("\n  Resumen:")
        # ↑ Imprime resumen de configuración antes de correr.
        print(f"    Eventos (intentos) : {self.n_events}")
        # ↑ Número de intentos Pythia.
        print(f"    pT min jets        : {self.jet_pt_min} GeV")
        # ↑ Corte de pT.
        print(f"    Hardness proxy     : {self.min_hard_parton_pt} GeV (0=off)")
        # ↑ Proxy de dureza.
        print(f"    Figuras jets       : {self.save_jet_figures}")
        # ↑ Flag de figuras.
        print(f"    Diagramas MG5      : {self.save_feynman_diagrams}")
        # ↑ Flag de diagramas.
        print("    Jet configs        :")
        # ↑ Encabezado del listado de configuraciones.
        for cfg in self.jet_configs:
            # ↑ Recorre configs seleccionadas.
            print(f'      R={cfg["R"]}  ->  {cfg["algo_name"]} (code={cfg["algo_code"]})')
            # ↑ Muestra radio, algoritmo y código asociado.

        ans = input("\n  ¿Continuar? (s/n): ").strip().lower()
        # ↑ Confirmación final antes de ejecutar el procesamiento pesado.

        return ans in {"s", "si", "sí", "y", "yes"}
        # ↑ Devuelve True si la respuesta fue afirmativa en español o inglés.

    # --------------------------
    # Procesamiento
    # --------------------------
    # ↑ Métodos internos del procesamiento físico (LHE, Pythia, FastJet, features).

    def _decompress_lhe_if_needed(self):
        # ↑ Descomprime el LHE si viene en formato .gz y actualiza self.lhe_file al .lhe.
        if self.lhe_file.endswith(".gz"):
            # ↑ Solo actúa si el archivo actual termina en .gz.
            out_path = self.lhe_file[:-3]
            # ↑ Construye la ruta descomprimida quitando ".gz".
            if os.path.exists(out_path):
                # ↑ Si ya existe el .lhe descomprimido, no lo rehace.
                print(f"  LHE ya descomprimido, usando: {out_path}")
            else:
                # ↑ Si no existe, lo descomprime.
                print(f"  Descomprimiendo LHE -> {out_path}")
                with gzip.open(self.lhe_file, "rb") as fi, open(out_path, "wb") as fo:
                    # ↑ Abre comprimido en lectura binaria y salida .lhe en escritura binaria.
                    shutil.copyfileobj(fi, fo)
                    # ↑ Copia el contenido descomprimido de una sola vez.
            self.lhe_file = out_path
            # ↑ Actualiza la ruta para que Pythia lea el .lhe (no .gz).

    def _init_pythia(self):
        # ↑ Configura e inicializa una instancia de Pythia para leer el LHE.
        p = pythia8.Pythia()
        # ↑ Crea objeto Pythia.

        p.readString("Beams:frameType = 4")
        # ↑ frameType=4 indica que la entrada viene de archivo externo (LHEF).

        p.readString(f"Beams:LHEF = {self.lhe_file}")
        # ↑ Se le pasa la ruta del archivo LHE a Pythia.

        p.readString("PartonLevel:MPI = on")
        # ↑ Activa multi-parton interactions (si aplica en tu setup/physics tune).
        p.readString("PartonLevel:ISR = on")
        # ↑ Activa initial-state radiation.
        p.readString("PartonLevel:FSR = on")
        # ↑ Activa final-state radiation.
        p.readString("HadronLevel:Hadronize = on")
        # ↑ Activa hadronización para pasar de partones a hadrones.

        # Semilla reproducible de Pythia
        # ↑ Configuración explícita de semilla para reproducibilidad.
        p.readString("Random:setSeed = on")
        # ↑ Indica que Pythia use semilla definida por el usuario.
        p.readString(f"Random:seed = {self.pythia_seed}")
        # ↑ Asigna la semilla (42 por defecto).

        p.readString("Print:quiet = on")
        # ↑ Reduce verbosidad de Pythia para no inundar la terminal.

        if not p.init():
            # ↑ Inicializa Pythia; si falla, suele ser por LHE corrupto o config incompatible.
            raise RuntimeError("Pythia8 no pudo inicializarse con el archivo LHE dado.")

        return p
        # ↑ Devuelve el objeto Pythia inicializado.

    def _extract_partons_for_matching(self, pythia):
        # ↑ Extrae partones relevantes (quarks/gluón) para asignar flavour a jets.
        """
        Extrae partones relevantes (quarks/gluón) evitando duplicados en cadenas
        de shower/decaimiento (si tiene hija del mismo sabor, se omite).

        Devuelve tuplas: (pid, eta, phi, pt)
        """
        # ↑ La idea es evitar contar dos veces el "mismo" partón a través del shower chain.

        partons = []
        # ↑ Lista donde se guardarán los partones útiles para matching.

        for i in range(pythia.event.size()):
            # ↑ Recorre todas las partículas del evento Pythia (incluye intermedias/finales).
            p = pythia.event[i]
            # ↑ Accede al objeto partícula i.

            pid_abs = abs(p.id())
            # ↑ ID absoluto (ignorando signo) para comparar con conjuntos de IDs.
            if pid_abs not in QUARK_GLUON_IDS_ABS:
                # ↑ Si no es quark ni gluón, no sirve para flavour matching.
                continue

            stat_abs = abs(p.status())
            # ↑ Status absoluto (Pythia usa signos según contexto).
            if stat_abs not in RELEVANT_STATUS_ABS:
                # ↑ Si el status no es de los que queremos, lo ignoramos.
                continue

            d1, d2 = p.daughter1(), p.daughter2()
            # ↑ Índices de la primera y última hija en la tabla del evento.
            has_same_flavour_daughter = False
            # ↑ Flag para detectar si el partón "continúa" en una hija del mismo sabor.

            if d1 > 0 and d2 >= d1:
                # ↑ Solo entra si hay rango válido de hijas.
                for di in range(d1, d2 + 1):
                    # ↑ Recorre hijas del partón.
                    if 0 <= di < pythia.event.size():
                        # ↑ Protección por seguridad contra índices fuera de rango.
                        dau = pythia.event[di]
                        # ↑ Objeto hija.
                        if abs(dau.id()) == pid_abs and abs(dau.status()) in RELEVANT_STATUS_ABS:
                            # ↑ Si hay hija con mismo sabor y status relevante,
                            #   consideramos que este partón es un ancestro y lo omitimos.
                            has_same_flavour_daughter = True
                            break
                            # ↑ Sale del loop de hijas al encontrar continuidad de sabor.

            if not has_same_flavour_daughter:
                # ↑ Solo guardamos partones "terminales" en ese sentido (sin hija relevante del mismo sabor).
                partons.append((int(p.id()), float(p.eta()), float(p.phi()), float(p.pT())))
                # ↑ Guardamos PID con signo, eta, phi, pT para matching posterior.

        return partons
        # ↑ Devuelve la lista de partones útiles del evento.

    def _event_hardness_proxy_pt(self, partons):
        # ↑ Calcula un proxy de dureza del evento usando el máximo pT entre partones relevantes.
        """
        Proxy de dureza del evento usando partones relevantes.
        Retorna max pT entre partones de matching.
        NO es exactamente PhaseSpace:pTHatMin de Pythia-only.
        """
        # ↑ Aclara que es aproximado (muy útil para filtrar eventos suaves).

        if not partons:
            # ↑ Si no hay partones de matching, el proxy se define como 0.
            return 0.0

        max_pt = 0.0
        # ↑ Inicializa el máximo acumulado.
        for item in partons:
            # ↑ Recorre cada tupla (pid, eta, phi, pt).
            if len(item) >= 4:
                # ↑ Seguridad por si la estructura cambia (aunque aquí siempre debería ser 4).
                pt = float(item[3])
                # ↑ Extrae el pT.
                if pt > max_pt:
                    # ↑ Actualiza el máximo si corresponde.
                    max_pt = pt

        return max_pt
        # ↑ Devuelve el pT máximo encontrado.

    def _compute_fractions(self, const_info):
        # ↑ Calcula fracciones de energía y contadores a partir de constituyentes de un jet.
        """
        const_info: lista de (PseudoJet_constituent, pid, is_charged)
        """
        # ↑ Cada constituyente trae su cuatro-momento (PseudoJet), PDG ID y bandera de carga.

        E_total = sum(cj.e() for cj, _, _ in const_info)
        # ↑ Suma energía total de constituyentes del jet (denominador para fracciones).
        if E_total <= 0.0:
            # ↑ Protección numérica (en teoría no debería pasar, pero mejor evitar división por cero).
            E_total = 1e-9

        E_chf = 0.0  # hadrónico cargado
        # ↑ Energía hadrónica cargada (ej. piones cargados, protones, etc.).
        E_nef = 0.0  # EM neutro (fotones)
        # ↑ Energía electromagnética neutra (principalmente fotones).
        E_nhf = 0.0  # hadrónico neutro
        # ↑ Energía hadrónica neutra (ej. neutrones, K0L, etc.).
        E_cef = 0.0  # EM cargado (electrones)
        # ↑ Energía electromagnética cargada (electrones/positrones).

        n_ch = 0
        # ↑ Contador de constituyentes cargados.
        n_neu = 0
        # ↑ Contador de constituyentes neutros.
        has_b = False
        # ↑ Flag si se detecta un hadrón B en el jet.
        has_c = False
        # ↑ Flag si se detecta un hadrón C en el jet.
        n_sv = 0
        # ↑ Proxy del número de "secondary vertices" (partículas de vida larga).
        muon_pt = 0.0
        # ↑ pT máximo de muón encontrado dentro del jet (proxy útil).

        for cj, pid, is_charged in const_info:
            # ↑ Recorre cada constituyente del jet.
            e = cj.e()
            # ↑ Energía del constituyente.
            apid = abs(pid)
            # ↑ PDG absoluto para clasificar tipo de partícula.

            if is_charged:
                # ↑ Rama de constituyentes cargados.
                n_ch += 1
                # ↑ Incrementa contador cargado.
                if apid == 11:
                    # ↑ Si es electrón/positrón (EM cargado)...
                    E_cef += e
                    # ↑ ...suma a la fracción EM cargada.
                elif apid == 13:
                    # ↑ Si es muón...
                    muon_pt = max(muon_pt, cj.pt())
                    # ↑ ...guarda el pT máximo de muón del jet.
                else:
                    # ↑ Otros cargados (piones/protones/kaones cargados...) se cuentan hadrónicos.
                    E_chf += e
                    # ↑ Suma a energía hadrónica cargada.
            else:
                # ↑ Rama de constituyentes neutros.
                n_neu += 1
                # ↑ Incrementa contador neutro.
                if apid == 22:
                    # ↑ Fotón = componente EM neutra.
                    E_nef += e
                    # ↑ Suma a energía EM neutra.
                else:
                    # ↑ Otros neutros se consideran hadrónicos neutros.
                    E_nhf += e
                    # ↑ Suma a energía hadrónica neutra.

            if pid in B_HADRON_IDS:
                # ↑ Si el PDG con signo está en la lista de hadrones B...
                has_b = True
                # ↑ Marca presencia de B hadrón (proxy b-tag fuerte).

            if pid in C_HADRON_IDS:
                # ↑ Si el PDG con signo está en la lista de hadrones C...
                has_c = True
                # ↑ Marca presencia de C hadrón (proxy c-tag fuerte).

            if apid in LONG_LIVED_IDS_ABS:
                # ↑ Si es partícula de vida larga (valor absoluto)...
                n_sv += 1
                # ↑ Incrementa proxy de vértices secundarios.

        return {
            # ↑ Devuelve todas las fracciones y contadores en un diccionario.
            "nef": float(np.clip(E_nef / E_total, 0.0, 1.0)),
            # ↑ Fracción EM neutra, recortada a [0,1] por seguridad numérica.
            "nhf": float(np.clip(E_nhf / E_total, 0.0, 1.0)),
            # ↑ Fracción hadrónica neutra.
            "cef": float(np.clip(E_cef / E_total, 0.0, 1.0)),
            # ↑ Fracción EM cargada.
            "chf": float(np.clip(E_chf / E_total, 0.0, 1.0)),
            # ↑ Fracción hadrónica cargada.
            "ncharged": int(n_ch),
            # ↑ Conteo de cargados.
            "nneutral": int(n_neu),
            # ↑ Conteo de neutros.
            "n_const": int(len(const_info)),
            # ↑ Número total de constituyentes.
            "has_b": bool(has_b),
            # ↑ Presencia de hadrón B.
            "has_c": bool(has_c),
            # ↑ Presencia de hadrón C.
            "n_sv": int(n_sv),
            # ↑ Número de partículas de vida larga (proxy SV).
            "muon_pt": float(muon_pt),
            # ↑ pT del muón más energético dentro del jet.
        }

    def _match_flavour(self, jet_eta, jet_phi, partons, R):
        # ↑ Asigna flavour al jet por matching geométrico con partones relevantes.
        """
        Matching con prioridad física:
          b > c > t > light quarks > gluón
        en dos etapas:
          1) cono estricto max(0.2, 0.4*R)
          2) cono completo R

        Devuelve PDG ID CON SIGNO.
        """
        # ↑ Usa prioridad física para elegir el mejor partón si hay varios cercanos.

        priority = {5: 4, 4: 3, 6: 2, 1: 1, 2: 1, 3: 1, 21: 0}
        # ↑ Mapa de prioridad: b es mayor, luego c, luego top, luego quarks ligeros, luego gluón.

        def best_in_cone(dr_max):
            # ↑ Función interna que busca el mejor partón dentro de un cono dado ΔR < dr_max.
            best_pid = 0
            # ↑ PID 0 significa "no match".
            best_pri = -1
            # ↑ Prioridad inicial menor que cualquiera válida.
            best_dr = float("inf")
            # ↑ Distancia inicial infinita para desempate.

            for item in partons:
                # ↑ Recorre lista de partones (pid, eta, phi, pt).
                pid, p_eta, p_phi = item[0], item[1], item[2]
                # ↑ Extrae PID y coordenadas angulares del partón.
                dphi = abs(p_phi - jet_phi)
                # ↑ Diferencia absoluta en phi.
                if dphi > np.pi:
                    # ↑ Corrige periodicidad angular: usa la ruta corta en el círculo.
                    dphi = 2 * np.pi - dphi
                dr = np.sqrt((p_eta - jet_eta) ** 2 + dphi ** 2)
                # ↑ Calcula distancia ΔR = sqrt((Δη)^2 + (Δφ)^2).
                pri = priority.get(abs(pid), 0)
                # ↑ Prioridad según sabor del partón; default 0 si no está en el mapa.

                if dr < dr_max and (pri > best_pri or (pri == best_pri and dr < best_dr)):
                    # ↑ Se actualiza si:
                    #   1) está dentro del cono, y
                    #   2) tiene mejor prioridad, o misma prioridad pero menor ΔR.
                    best_pid = pid
                    # ↑ Guarda PID con signo.
                    best_pri = pri
                    # ↑ Guarda prioridad.
                    best_dr = dr
                    # ↑ Guarda distancia para desempate futuro.

            return int(best_pid)
            # ↑ Devuelve el PID del mejor match en ese cono (o 0 si ninguno).

        strict_cone = max(0.2, 0.4 * R)
        # ↑ Primer cono "estricto": no menor que 0.2, y escalable con R.

        pid_match = best_in_cone(strict_cone)
        # ↑ Intenta matching primero en cono estricto.
        if pid_match == 0:
            # ↑ Si no encontró nada...
            pid_match = best_in_cone(R)
            # ↑ ...relaja al cono completo del jet.

        return int(pid_match)
        # ↑ Devuelve PID con signo (antipartículas quedan negativas).

    def _apply_detector_smearing(self, pt_true, eta_true, phi_true, m_true):
        # ↑ Aplica smearing paramétrico para generar variables "reco-like" a partir de GEN.
        """
        Smearing paramétrico tipo CMS/ATLAS usando self.rng (reproducible).
        """
        # ↑ Modelo simplificado, útil para ML/prototipos sin simulación Geant detallada.

        a = 1.0
        # ↑ Término dominante ~1/sqrt(pt) en resolución de pT (parámetro toy).
        b = 0.05
        # ↑ Término constante de resolución (parámetro toy).

        sigma_pt = pt_true * np.sqrt((a / np.sqrt(max(pt_true, 1.0))) ** 2 + b ** 2)
        # ↑ Resolución de pT tipo cuadratura:
        #   sigma/pt = sqrt((a/sqrt(pt))^2 + b^2)

        reco_pt = max(0.0, self.rng.normal(pt_true, sigma_pt))
        # ↑ Muestra pT reconstruido de una normal; se recorta a >= 0.
        reco_eta = float(self.rng.normal(eta_true, 0.01))
        # ↑ Smearing gaussiano en eta con sigma fija.
        reco_phi = wrap_phi(self.rng.normal(phi_true, 0.01))
        # ↑ Smearing gaussiano en phi y luego se envuelve a (-pi, pi].
        reco_m = max(0.0, self.rng.normal(m_true, 0.05 * max(m_true, 0.1)))
        # ↑ Smearing en masa con sigma proporcional a la masa (evitando sigma cero si m~0).

        return float(reco_pt), reco_eta, reco_phi, float(reco_m)
        # ↑ Devuelve las cuatro variables reco-like.

    def _compute_btag(self, fracs, flavour):
        # ↑ Calcula un score proxy de b-tag a partir de hadrones B/C y flavour asignado.
        if fracs["has_b"] or abs(flavour) == 5:
            # ↑ Si hay evidencia fuerte de jet b (hadrones B o flavour=±5)...
            return float(np.clip(self.rng.normal(0.85, 0.10), 0.0, 1.0))
            # ↑ ...score alto, con dispersión realista toy.

        if fracs["has_c"] or abs(flavour) == 4:
            # ↑ Si parece jet c...
            return float(np.clip(self.rng.normal(0.25, 0.10), 0.0, 1.0))
            # ↑ ...score intermedio (mistag parcial).

        return float(np.clip(self.rng.exponential(0.05), 0.0, 0.30))
        # ↑ Jets light/gluon: score bajo, modelado con exponencial (cola pequeña).

    def _compute_ctag(self, fracs, flavour):
        # ↑ Calcula score proxy de c-tag (lógica análoga a btag pero con prioridades distintas).
        if fracs["has_c"] or abs(flavour) == 4:
            # ↑ Si el jet tiene señales de charm...
            return float(np.clip(self.rng.normal(0.80, 0.12), 0.0, 1.0))
            # ↑ ...score alto de c-tag.

        if fracs["has_b"] or abs(flavour) == 5:
            # ↑ Si es b-jet, puede haber algo de c-tag (mistag / cascadas).
            return float(np.clip(self.rng.normal(0.15, 0.08), 0.0, 1.0))
            # ↑ ...score bajo-moderado.

        return float(np.clip(self.rng.exponential(0.04), 0.0, 0.25))
        # ↑ Jets light/gluon: c-tag bajo.

    def _print_sanity(self, dataset, key):
        # ↑ Imprime chequeos rápidos de sanidad física del dataset de jets.
        """
        Chequeos rápidos de sanidad física.
        dataset debe tener shape [N, 24].
        """
        # ↑ Se asume dataset 2D con 24 columnas.

        if dataset.shape[0] == 0:
            # ↑ Si no hay jets, solo reporta y sale.
            print(f"    Sanity {key}: dataset vacío (0 jets).")
            return

        frac_sum = dataset[:, 11] + dataset[:, 12] + dataset[:, 13] + dataset[:, 14]
        # ↑ Suma NEF + NHF + CEF + CHF (idealmente ~1 si las fracciones están bien construidas).

        bad_frac = np.mean((frac_sum < 0.5) | (frac_sum > 1.5))
        # ↑ Fracción de jets con suma de fracciones sospechosa (muy lejos de 1).

        ratio = np.median(dataset[:, 6] / np.clip(dataset[:, 0], 1e-6, None))
        # ↑ Mediana de recoPt/genPt (debería estar cerca de 1 con smearing razonable).

        if dataset.shape[0] < 2 or np.std(dataset[:, 0]) == 0 or np.std(dataset[:, 6]) == 0:
            # ↑ Si hay muy pocos jets o pT sin varianza, la correlación no está definida.
            corr = np.nan
            # ↑ Se usa NaN para indicar que no aplica.
        else:
            corr = np.corrcoef(dataset[:, 0], dataset[:, 6])[0, 1]
            # ↑ Correlación lineal entre pT gen y reco; debería ser alta.

        print(f"    Sanity {key}:")
        # ↑ Encabezado de sanity checks.
        print(f"      pT corr gen/reco     : {corr:.4f}  (esperado > 0.95, si hay suficientes jets)")
        # ↑ Correlación alta indica que el smearing no rompió la estructura física.
        print(f"      pT ratio mediana     : {ratio:.3f}  (esperado ~1.0)")
        # ↑ ratio ~1 sugiere smearing centrado.
        print(f"      Jets con fracs raras : {100 * bad_frac:.1f}%  (esperado bajo)")
        # ↑ Un porcentaje alto aquí puede indicar bug en fracciones.
        print(f"      Fracs EM+HAD mediana : {np.median(frac_sum):.3f}  (esperado ~1.0)")
        # ↑ La mediana de la suma de fracciones debe rondar 1.

    # --------------------------
    # Figuras
    # --------------------------
    # ↑ Métodos para generar histogramas y scatter plots de jets.

    def _plot_event_jets_eta_phi_from_arrays(self, jets_eta_phi_pt, cfg_key, source_event_idx, accepted_event_idx, out_dir):
        # Wrapper: implementado en módulo externo para mantener este archivo más corto.
        return plot_event_jets_eta_phi_from_arrays(jets_eta_phi_pt, cfg_key, source_event_idx, accepted_event_idx, out_dir)

    def _plot_global_dataset_figures(self, dataset, cfg_key, out_dir):
        # Wrapper: implementado en módulo externo para mantener este archivo más corto.
        return plot_global_dataset_figures(self, dataset, cfg_key, out_dir)

    def _convert_ps_eps_to_pdf_jpg(self, src_path, out_dir):
        # Wrapper: implementado en módulo externo para mantener este archivo más corto.
        return convert_ps_eps_to_pdf_jpg(self, src_path, out_dir)

    def _collect_feynman_diagrams(self, run_dir):
        # Wrapper: implementado en módulo externo para mantener este archivo más corto.
        return collect_feynman_diagrams(self, run_dir)

    def process_with_pythia_fastjet(self):
        # ↑ Ejecuta el paso 3 del workflow y devuelve datasets + número de eventos aceptados.
        print("\n[3/4] PYTHIA + FASTJET")
        # ↑ Título del paso.
        print("-" * 50)
        # ↑ Separador visual.

        self._decompress_lhe_if_needed()
        # ↑ Se asegura de que el LHE esté descomprimido antes de inicializar Pythia.

        print("  Inicializando Pythia8...")
        # ↑ Mensaje de progreso.
        pythia = self._init_pythia()
        # ↑ Crea e inicializa Pythia con el LHE actual.

        print("  Leyendo eventos de Pythia...")
        # ↑ Mensaje de progreso.
        stored_events = []
        # ↑ Lista donde se almacenan eventos "aceptados" ya convertidos a estructuras útiles.

        # Cada entrada:
        # {
        #   "source_event_idx": int,
        #   "accepted_event_idx": int,
        #   "particles": [(PseudoJet, pid, is_charged), ...],
        #   "particle_map": {user_index: (pid, is_charged)},
        #   "partons": [(pid, eta, phi, pt), ...],
        #   "hard_proxy_pt": float
        # }
        # ↑ Estructura esperada por evento para poder reusar el mismo evento en varias configs de jets.

        accepted_counter = 0
        # ↑ Contador de eventos aceptados (puede ser menor que n_events si aplicas hardness proxy).

        for i_ev in range(self.n_events):
            # ↑ Loop principal sobre intentos de eventos Pythia.
            if not pythia.next():
                # ↑ Pide siguiente evento; si no hay más (EOF o fallo), termina.
                print(f"  ⚠ Pythia terminó en intento {i_ev}")
                break

            particles = []
            # ↑ Lista de partículas finales visibles del evento, en formato útil para FastJet.
            particle_map = {}
            # ↑ Mapa user_index -> (pid, is_charged) para recuperar info de constituyentes luego.

            for i in range(pythia.event.size()):
                # ↑ Recorre toda la tabla del evento Pythia.
                p = pythia.event[i]
                # ↑ Partícula i del evento.

                if not p.isFinal():
                    # ↑ Solo se clusterizan partículas finales (estado final).
                    continue

                # Excluir neutrinos (jets visibles/reco-like)
                # ↑ Neutrinos no deberían entrar en jets reconstruidos "visibles".
                if abs(p.id()) in NEUTRINO_IDS_ABS:
                    # ↑ Si es neutrino νe/νμ/ντ, se omite.
                    continue

                px, py, pz, e = p.px(), p.py(), p.pz(), p.e()
                # ↑ Extrae componentes del 4-momento en convención cartesianas + energía.

                if (
                    not np.isfinite(px) or not np.isfinite(py) or
                    not np.isfinite(pz) or not np.isfinite(e)
                ):
                    # ↑ Protección contra NaN/inf por cualquier rareza numérica.
                    continue

                if e <= 0.0:
                    # ↑ FastJet requiere energías físicas positivas.
                    continue

                pj = fj.PseudoJet(px, py, pz, e)
                # ↑ Construye PseudoJet de FastJet para esta partícula final.
                pj.set_user_index(i)
                # ↑ Guarda el índice de Pythia dentro del PseudoJet para mapear constituyentes luego.

                pid = int(p.id())
                # ↑ PDG ID con signo (importantísimo para flavour/hadrones B/C).
                is_charged = bool(p.isCharged())
                # ↑ Bandera de carga (la da Pythia).

                particles.append((pj, pid, is_charged))
                # ↑ Guarda la partícula visible en formato de trabajo.
                particle_map[i] = (pid, is_charged)
                # ↑ Mapa por user_index para recuperar PID/carga desde constituyentes de FastJet.

            partons = self._extract_partons_for_matching(pythia)
            # ↑ Extrae partones relevantes del mismo evento para flavour matching.
            hard_proxy_pt = self._event_hardness_proxy_pt(partons)
            # ↑ Calcula proxy de dureza (max pT partón relevante).

            # Filtro tipo "pTHat" proxy (solo si está activado)
            # ↑ Si el usuario activó min_hard_parton_pt, se filtran eventos "blandos".
            if self.min_hard_parton_pt > 0.0 and hard_proxy_pt < self.min_hard_parton_pt:
                # ↑ Si el evento no alcanza el umbral de dureza, se descarta.
                continue

            stored_events.append({
                "source_event_idx": int(i_ev),
                # ↑ Índice del intento de evento dentro del loop Pythia.
                "accepted_event_idx": int(accepted_counter),
                # ↑ Índice corrido solo de eventos aceptados.
                "particles": particles,
                # ↑ Lista de partículas finales visibles (PseudoJet + PID + carga).
                "particle_map": particle_map,
                # ↑ Mapa para reconstruir info de constituyentes.
                "partons": partons,
                # ↑ Partones relevantes para flavour matching.
                "hard_proxy_pt": float(hard_proxy_pt),
                # ↑ Proxy de dureza guardado para referencia/debug.
            })
            accepted_counter += 1
            # ↑ Incrementa contador de aceptados.

            if (i_ev + 1) % 200 == 0:
                # ↑ Imprime progreso cada 200 intentos (útil para runs grandes).
                print(f"    {i_ev + 1}/{self.n_events} intentos leídos...  aceptados: {accepted_counter}")

        n_events_real = len(stored_events)
        # ↑ Número real de eventos aceptados y almacenados.
        print(f"  ✓ {n_events_real} eventos aceptados y almacenados")
        # ↑ Confirmación.
        if self.min_hard_parton_pt > 0:
            # ↑ Solo imprime detalle del filtro si estaba activo.
            print(f"    (Con hardness proxy >= {self.min_hard_parton_pt} GeV)")

        all_datasets = {}
        # ↑ Diccionario con un dataset por configuración de jet.

        # all_datasets[key] = {
        #   "data": np.ndarray,
        #   "algo": str,
        #   "algo_code": int,
        #   "R": float,
        #   "event_figures": [...]
        # }
        # ↑ Estructura de salida por configuración.

        for cfg in self.jet_configs:
            # ↑ Recorre cada configuración de jet (algoritmo + R).
            algo_name = cfg["algo_name"]
            # ↑ Nombre del algoritmo.
            algo_func = cfg["algo_func"]
            # ↑ Función/enum de FastJet.
            algo_code = int(cfg["algo_code"])
            # ↑ Código numérico para guardar en feature 23.
            R = float(cfg["R"])
            # ↑ Radio del jet.

            key = f"{algo_name}_R{R:g}"
            # ↑ Clave legible para esta configuración (ej. "antikt_R0.4").
            print(f"\n  Procesando {key} (pT_min={self.jet_pt_min} GeV)...")
            # ↑ Log de progreso.

            jet_def = fj.JetDefinition(algo_func, R)
            # ↑ Define el algoritmo de clustering en FastJet con ese radio.
            dataset_rows = []
            # ↑ Lista de filas (features de jets) para esta configuración.
            event_figures = []
            # ↑ Lista de info mínima para generar figuras por evento después.

            for event_data in stored_events:
                # ↑ Reutiliza los eventos ya almacenados para cada config (eficiente).
                particles = event_data["particles"]
                # ↑ Partículas visibles del evento.
                particle_map = event_data["particle_map"]
                # ↑ Mapa user_index -> (pid, charge).
                partons = event_data["partons"]
                # ↑ Partones para flavour matching.

                pseudojets = [pj for pj, _, _ in particles]
                # ↑ Extrae solo los PseudoJet (FastJet) del evento.
                if not pseudojets:
                    # ↑ Si el evento quedó vacío tras filtros, se omite.
                    continue

                cs = fj.ClusterSequence(pseudojets, jet_def)
                # ↑ Ejecuta el clustering de FastJet con la definición actual.

                try:
                    jets_all = fj.sorted_by_pt(cs.inclusive_jets())
                    # ↑ Obtiene todos los jets inclusivos y los ordena por pT descendente.
                except Exception:
                    # ↑ Fallback si la función wrapper falla en alguna versión.
                    jets_all = sorted(cs.inclusive_jets(), key=lambda j: -j.pt())
                    # ↑ Ordena manualmente por pT descendente.

                jets = [j for j in jets_all if j.pt() >= self.jet_pt_min]
                # ↑ Aplica corte de pT mínimo a los jets.

                if not jets:
                    # ↑ Si ningún jet pasa el corte, sigue al próximo evento.
                    continue

                if self.save_jet_figures and len(event_figures) < self.max_event_figures_per_cfg:
                    # ↑ Guarda info de este evento para scatter por evento (limitado por cfg).
                    jets_eta_phi_pt = [(float(j.eta()), float(j.phi()), float(j.pt())) for j in jets]
                    # ↑ Extrae eta, phi, pt de cada jet (para plotting posterior).
                    event_figures.append({
                        "source_event_idx": int(event_data["source_event_idx"]),
                        # ↑ Índice del intento original.
                        "accepted_event_idx": int(event_data["accepted_event_idx"]),
                        # ↑ Índice del evento aceptado.
                        "jets_eta_phi_pt": jets_eta_phi_pt,
                        # ↑ Jets del evento en coordenadas angulares + pT.
                    })

                for jet in jets:
                    # ↑ Recorre cada jet del evento para calcular features.
                    const_info = []
                    # ↑ Lista de constituyentes del jet con su PID/carga.
                    for cj in jet.constituents():
                        # ↑ FastJet devuelve constituyentes como PseudoJets.
                        idx = cj.user_index()
                        # ↑ Recupera el índice original de Pythia (guardado antes).
                        info = particle_map.get(idx)
                        # ↑ Busca (pid, is_charged) en el mapa.
                        if info is None:
                            # ↑ Si no aparece (raro), se omite ese constituyente.
                            continue
                        pid, is_charged = info
                        # ↑ Desempaqueta PID y carga.
                        const_info.append((cj, pid, is_charged))
                        # ↑ Agrega constituyente enriquecido a la lista.

                    if not const_info:
                        # ↑ Si por alguna razón no quedó info de constituyentes, no se puede seguir.
                        continue

                    fracs = self._compute_fractions(const_info)
                    # ↑ Calcula fracciones de energía, ncharged, nneutral, has_b, has_c, etc.
                    flavour = self._match_flavour(jet.eta(), jet.phi(), partons, R)
                    # ↑ Asigna flavour (PDG con signo) por matching con partones.
                    rPt, rEta, rPhi, rM = self._apply_detector_smearing(
                        jet.pt(), jet.eta(), jet.phi(), jet.m()
                    )
                    # ↑ Genera variables reconstruidas (reco-like) por smearing.

                    jet_id = jet_quality_id(fracs)
                    # ↑ Calcula JetID proxy (0/1/3).
                    qgl = quark_gluon_likelihood(fracs)
                    # ↑ Calcula QGL proxy en [0,1].

                    btag = self._compute_btag(fracs, flavour)
                    # ↑ Score de b-tag proxy.
                    ctag = self._compute_ctag(fracs, flavour)
                    # ↑ Score de c-tag proxy.

                    feat = np.array([
                        jet.pt(),                   # 0  pt_gen
                        # ↑ pT "GEN-like" del jet clustered.
                        jet.eta(),                  # 1  eta_gen
                        # ↑ eta del jet.
                        wrap_phi(jet.phi()),        # 2  phi_gen
                        # ↑ CORRECCIÓN adicional:
                        #   Guardamos phi envuelto explícitamente en (-pi, pi] para consistencia global.
                        jet.m(),                    # 3  m_gen
                        # ↑ Masa del jet.
                        float(flavour),             # 4  flavour (PDG con signo)
                        # ↑ PDG ID con signo, guardado como float32 para compatibilidad con array homogéneo.
                        btag,                       # 5
                        # ↑ b-tag proxy.
                        rPt,                        # 6  recoPt
                        # ↑ pT reconstruido (smearing).
                        rPhi,                       # 7  recoPhi
                        # ↑ phi reconstruido (ya envuelto por _apply_detector_smearing).
                        rEta,                       # 8  recoEta
                        # ↑ eta reconstruido.
                        fracs["muon_pt"],           # 9
                        # ↑ pT máximo de muón dentro del jet (proxy).
                        float(fracs["n_const"]),    # 10
                        # ↑ Número de constituyentes (float para mantener dtype único).
                        fracs["nef"],               # 11
                        # ↑ Neutral EM fraction.
                        fracs["nhf"],               # 12
                        # ↑ Neutral hadronic fraction.
                        fracs["cef"],               # 13
                        # ↑ Charged EM fraction.
                        fracs["chf"],               # 14
                        # ↑ Charged hadronic fraction.
                        qgl,                        # 15
                        # ↑ QGL proxy.
                        float(jet_id),              # 16
                        # ↑ JetID proxy (0/1/3) como float.
                        float(fracs["ncharged"]),   # 17
                        # ↑ Número de constituyentes cargados.
                        float(fracs["nneutral"]),   # 18
                        # ↑ Número de constituyentes neutros.
                        ctag,                       # 19
                        # ↑ c-tag proxy.
                        float(fracs["n_sv"]),       # 20
                        # ↑ Proxy de secondary vertices.
                        rM,                         # 21 recoMass
                        # ↑ Masa reconstruida.
                        float(R),                   # 22 jetR
                        # ↑ Radio del jet usado para esa fila.
                        float(algo_code),           # 23 algoCode
                        # ↑ Código del algoritmo (1/2/3).
                    ], dtype=np.float32)
                    # ↑ Convierte toda la fila a float32 para compactar almacenamiento.

                    dataset_rows.append(feat)
                    # ↑ Agrega la fila del jet al dataset de esta configuración.

            if dataset_rows:
                # ↑ Si se recolectaron jets para esta config...
                dataset = np.asarray(dataset_rows, dtype=np.float32)
                # ↑ Convierte lista de filas a array 2D.
            else:
                # ↑ Si no hubo jets, crea array vacío con shape correcto.
                dataset = np.empty((0, N_FEATURES), dtype=np.float32)

            all_datasets[key] = {
                "data": dataset,
                # ↑ Dataset numérico [N_jets, 24].
                "algo": algo_name,
                # ↑ Nombre del algoritmo.
                "algo_code": algo_code,
                # ↑ Código del algoritmo.
                "R": R,
                # ↑ Radio usado.
                "event_figures": event_figures,
                # ↑ Info de eventos para plots por evento.
            }

            print(f"    -> {dataset.shape[0]} jets ({key})")
            # ↑ Reporta cuántos jets produjo esta configuración.
            self._print_sanity(dataset, key)
            # ↑ Imprime chequeos de sanidad para esta configuración.

        try:
            pythia.stat()
            # ↑ Muestra estadísticas internas de Pythia (cross sections, counters, etc.).
        except Exception:
            # ↑ Si el wrapper falla en alguna instalación, no rompe el flujo.
            pass

        return all_datasets, n_events_real
        # ↑ Devuelve datasets por configuración y número de eventos aceptados.

    # --------------------------
    # Guardado
    # --------------------------
    # ↑ Métodos para guardar .npy, metadata, preview, figuras y README.

    def save_datasets(self, all_datasets, n_events):
        # Wrapper: implementado en módulo externo para mantener este archivo más corto.
        return save_datasets(self, all_datasets, n_events)

    def diagnose_parton_status(self, n_events_diag=5):
        # ↑ Inspecciona status de quarks/gluones en eventos Pythia para ajustar matching.
        """
        Útil si la columna flavour sale muy llena de ceros.
        """
        # ↑ Te ayuda a validar si RELEVANT_STATUS_ABS está bien para tu muestra/proceso.

        if not self.lhe_file:
            # ↑ Requiere que ya exista un LHE cargado.
            raise ValueError("Primero define self.lhe_file (o corre el workflow hasta cargar el LHE).")

        print("\nDIAGNÓSTICO DE STATUS DE PARTONES")
        # ↑ Encabezado del diagnóstico.
        print("-" * 50)
        # ↑ Separador.

        self._decompress_lhe_if_needed()
        # ↑ Se asegura de trabajar con .lhe descomprimido.

        pythia = pythia8.Pythia()
        # ↑ Crea nueva instancia separada de Pythia para el diagnóstico.
        pythia.readString("Beams:frameType = 4")
        # ↑ Entrada desde LHE.
        pythia.readString(f"Beams:LHEF = {self.lhe_file}")
        # ↑ Ruta al LHE.
        pythia.readString("HadronLevel:Hadronize = on")
        # ↑ Hadronización activada (para conservar flujo usual de evento).
        pythia.readString("Print:quiet = on")
        # ↑ Menos verbosidad.
        if not pythia.init():
            # ↑ Si falla init, no se puede hacer el diagnóstico.
            raise RuntimeError("No se pudo inicializar Pythia para diagnóstico.")

        status_counts = {}
        # ↑ Diccionario status -> conteo.

        for _ in range(n_events_diag):
            # ↑ Recorre unos pocos eventos (configurable) para inspección rápida.
            if not pythia.next():
                # ↑ Si se acaban/fallan los eventos, se corta.
                break
            for i in range(pythia.event.size()):
                # ↑ Recorre partículas del evento.
                p = pythia.event[i]
                # ↑ Accede a la partícula i.
                if abs(p.id()) in QUARK_GLUON_IDS_ABS:
                    # ↑ Solo interesa quarks y gluones.
                    s = int(p.status())
                    # ↑ Status con signo (aquí se reporta tal cual para ver todo).
                    status_counts[s] = status_counts.get(s, 0) + 1
                    # ↑ Acumula conteo por status.

        print("  Status encontrados (quarks/gluón):")
        # ↑ Encabezado de resultados.
        for s, c in sorted(status_counts.items()):
            # ↑ Muestra statuses ordenados numéricamente.
            bar = "█" * min(30, c)
            # ↑ Barrita visual simple (truncada) para intuir magnitud.
            print(f"    status {s:+4d}: {c:5d}  {bar}")
            # ↑ Imprime status, conteo y barra.

        print("\n  Recuerda: el matching usa abs(status) en:")
        # ↑ Recordatorio para interpretar la tabla (se usa valor absoluto).
        print(f"    {sorted(RELEVANT_STATUS_ABS)}")
        # ↑ Lista de statuses que actualmente considera el matching.

    # --------------------------
    # Flujo principal
    # --------------------------
    # ↑ Método de alto nivel que ejecuta todo el workflow en orden.

    def run(self):
        # ↑ Orquestador principal del script.
        self.print_header()
        # ↑ Paso visual: imprime cabecera.
        self.run_madgraph_interactive()
        # ↑ Paso 1: abre MG5 y localiza LHE.

        if not self.configure_analysis():
            # ↑ Paso 2: pide config; si el usuario cancela...
            print("Análisis cancelado.")
            # ↑ ...avisa y termina el flujo.
            return

        all_datasets, n_ev = self.process_with_pythia_fastjet()
        # ↑ Paso 3: corre Pythia+FastJet y arma datasets.
        run_dir, saved = self.save_datasets(all_datasets, n_ev)
        # ↑ Paso 4: guarda todo en disco.

        print("\n" + "=" * 80)
        # ↑ Línea decorativa de cierre.
        print("  WORKFLOW COMPLETADO")
        # ↑ Mensaje principal de éxito.
        print("=" * 80)
        # ↑ Línea decorativa.
        print(f"  Proceso   : {self.process_name}")
        # ↑ Nombre del proceso.
        print(f"  Eventos   : {n_ev} (aceptados)")
        # ↑ Número de eventos aceptados.
        print(f"  pT min    : {self.jet_pt_min} GeV")
        # ↑ Corte de pT usado.
        print(f"  Hard proxy: {self.min_hard_parton_pt} GeV (0=off)")
        # ↑ Umbral de dureza proxy usado.
        print(f"  Carpeta   : {run_dir}")
        # ↑ Carpeta final con todos los resultados.
        print()
        # ↑ Línea en blanco.

        for cfg_key, payload in all_datasets.items():
            # ↑ Resumen por configuración.
            arr = payload["data"]
            # ↑ Dataset de esa config.
            print(f"  {cfg_key:25s} -> {arr.shape[0]:6d} jets")
            # ↑ Muestra cantidad de jets.

        print("\n  Archivos .npy:")
        # ↑ Encabezado lista de arrays guardados.
        for path in saved:
            # ↑ Recorre rutas guardadas.
            print(f"    {path}")
            # ↑ Imprime ruta de cada .npy.
