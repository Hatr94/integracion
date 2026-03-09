#!/usr/bin/env python3
"""Entry point del pipeline.

Ejecuta el workflow completo: MadGraph -> Pythia8 -> FastJet.

Toda la configuración (eventos, jets, pileup) se hace de forma
interactiva al correr. No hay nada que editar aquí.

Uso:
    cd ~/integracion
    python -m hep_pipeline_pileup.main

Nota (referencia paper PUMML):
    - Entrenamiento: NPU ~ Poisson(mean=140), recortado a [0, 180]
    - PUPPI defaults: R0=0.3, Rmin=0.02, wcut=0.1, ptcut=0.1+0.007*NPU
"""

from hep_pipeline_pileup.workflow_core_pileup import HEPWorkflow


def main():
    workflow = HEPWorkflow()
    try:
        workflow.run()
    except KeyboardInterrupt:
        print("\n\nWorkflow interrumpido por el usuario.")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
