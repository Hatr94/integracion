#!/usr/bin/env python3
"""Pipeline entry point.

Run the full workflow.
"""

from hep_pipeline.workflow_core import HEPWorkflow

def main():
    workflow = HEPWorkflow()
    try:
        workflow.run()
    except KeyboardInterrupt:
        print("\n\nWorkflow interrupted por el usuario.")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
