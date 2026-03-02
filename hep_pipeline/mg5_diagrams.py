"""Helpers to collect and convert MadGraph diagram files when available."""

import os
import glob
import shutil
import subprocess

def collect_feynman_diagrams(workflow, run_dir):
    """
    Busca diagramas de MG5 y los copia a run_dir/feynman_diagrams.
    Ademas converts .ps/.eps a .pdf/.jpg si es posible.
    """

    if not workflow.process_name:
        return

    proc_candidates = [
        os.path.join(workflow.work_dir, workflow.process_name),
        os.path.join(workflow.work_dir, os.path.basename(workflow.process_name)),
    ]

    proc_dir = None
    for c in proc_candidates:
        if os.path.isdir(c):
            proc_dir = c
            break

    if proc_dir is None:
        print("  [diag] Could not find the MG5 process folder to collect diagrams.")
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
        print("  [diag] No diagrams were found in common formats.")
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

            made = workflow._convert_ps_eps_to_pdf_jpg(dst, out_dir)
            converted += len(made)

        except Exception as e:
            print(f"  [diag] Could not copy {src}: {e}")

    print(f"  [diag] Diagramas copiados a: {out_dir} ({copied} archivos, {converted} conversiones)")


def convert_ps_eps_to_pdf_jpg(workflow, src_path, out_dir):
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
    convert_bin = shutil.which("convert")

    pdf_path = os.path.join(out_dir, f"{base_name}.pdf")
    if ps2pdf_bin:
        try:
            subprocess.run([ps2pdf_bin, src_path, pdf_path], check=True)
            if os.path.exists(pdf_path):
                created.append(pdf_path)
        except Exception as e:
            print(f"  [diag] Error convirtiendo a PDF ({src_path}): {e}")
    else:
        print("  [diag] 'ps2pdf' is not installed; could not generate PDF.")

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
        print("  [diag] 'magick' or 'convert' is not installed; could not generate JPG.")

    return created
