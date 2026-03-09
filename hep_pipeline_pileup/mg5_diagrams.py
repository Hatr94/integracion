"""Utilidades para copiar/convertir diagramas de MadGraph (si existen)."""

import os
import glob
import shutil
import subprocess

def collect_feynman_diagrams(workflow, run_dir):
    # ↑ Busca diagramas en la carpeta del proceso MG5, los copia al run y convierte PS/EPS.
    """
    Busca diagramas de MG5 y los copia a run_dir/feynman_diagrams.
    Además convierte .ps/.eps a .pdf/.jpg si es posible.
    """
    # ↑ Esto ayuda a dejar todo el material de un run en una sola carpeta.

    if not workflow.process_name:
        # ↑ Si no hay nombre de proceso, no se puede ubicar la carpeta MG5.
        return

    proc_candidates = [
        os.path.join(workflow.work_dir, workflow.process_name),
        # ↑ Candidato típico: ./<process_name>
        os.path.join(workflow.work_dir, os.path.basename(workflow.process_name)),
        # ↑ Candidato redundante/seguro por si process_name vino como ruta.
    ]

    proc_dir = None
    # ↑ Aquí se guardará la ruta real al directorio del proceso MG5.
    for c in proc_candidates:
        # ↑ Revisa candidatos.
        if os.path.isdir(c):
            # ↑ Si existe y es carpeta...
            proc_dir = c
            # ↑ ...esa es la carpeta del proceso.
            break
            # ↑ Sale del loop al encontrar la primera válida.

    if proc_dir is None:
        # ↑ Si no se localizó la carpeta del proceso, avisa y termina.
        print("  [diag] No se encontró carpeta del proceso MG5 para recopilar diagramas.")
        return

    out_dir = os.path.join(run_dir, "feynman_diagrams")
    # ↑ Carpeta dentro del run donde se guardarán los diagramas.
    os.makedirs(out_dir, exist_ok=True)
    # ↑ La crea si no existe.

    patterns = [
        # ↑ Patrones de búsqueda en subcarpetas comunes de MG5.
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
    # ↑ Se buscan varios formatos en SubProcesses y HTML porque MG5 los puede dejar en ambos.

    found = []
    # ↑ Acumulador de archivos encontrados.
    for pat in patterns:
        # ↑ Recorre cada patrón.
        found.extend(glob.glob(pat, recursive=True))
        # ↑ Agrega resultados de búsqueda recursiva.

    found = sorted(set(found))
    # ↑ Elimina duplicados y ordena para reproducibilidad.

    if not found:
        # ↑ Si no encontró ningún diagrama en formatos comunes, avisa y termina.
        print("  [diag] No se encontraron diagramas en formatos comunes.")
        return

    copied = 0
    # ↑ Contador de archivos copiados.
    converted = 0
    # ↑ Contador de archivos convertidos (PDF/JPG generados).

    for src in found:
        # ↑ Copia cada archivo encontrado.
        name = os.path.basename(src)
        # ↑ Nombre del archivo.
        dst = os.path.join(out_dir, name)
        # ↑ Ruta destino en la carpeta del run.

        if os.path.exists(dst):
            # ↑ Si ya existe un archivo con ese nombre, evita sobrescribirlo.
            base, ext = os.path.splitext(name)
            # ↑ Separa nombre base y extensión.
            k = 1
            # ↑ Sufijo incremental para nombre alterno.
            while True:
                # ↑ Busca un nombre libre.
                dst_try = os.path.join(out_dir, f"{base}_{k}{ext}")
                # ↑ Propuesta de nombre con sufijo.
                if not os.path.exists(dst_try):
                    # ↑ Si no existe, se usa ese.
                    dst = dst_try
                    break
                k += 1
                # ↑ Si existe, incrementa sufijo y prueba otra vez.

        try:
            shutil.copy2(src, dst)
            # ↑ Copia preservando metadatos (mtime, etc.).
            copied += 1
            # ↑ Incrementa contador de copiados.

            # Si es .ps o .eps, intentar convertir a PDF/JPG
            # ↑ Además de copiar, intenta conversión para formatos vectoriales clásicos.
            made = workflow._convert_ps_eps_to_pdf_jpg(dst, out_dir)
            # ↑ Devuelve lista de archivos creados.
            converted += len(made)
            # ↑ Suma cuántos nuevos archivos de conversión se generaron.

        except Exception as e:
            # ↑ Si falla la copia de un archivo, se reporta pero se sigue con los demás.
            print(f"  [diag] No pude copiar {src}: {e}")

    print(f"  [diag] Diagramas copiados a: {out_dir} ({copied} archivos, {converted} conversiones)")
    # ↑ Resumen final del paso de diagramas.

# --------------------------
# Procesamiento principal
# --------------------------
# ↑ Método central: lee LHE -> Pythia -> FastJet -> arma datasets por configuración.

def convert_ps_eps_to_pdf_jpg(workflow, src_path, out_dir):
    # ↑ Convierte un archivo .ps/.eps a .pdf y .jpg si hay herramientas disponibles.
    """
    Convierte .ps/.eps a .pdf y .jpg si hay herramientas disponibles.
    Requiere:
      - ps2pdf (Ghostscript)
      - magick (ImageMagick) o convert
    """
    # ↑ Estas conversiones son útiles porque MG5 a veces guarda diagramas en .ps/.eps.

    ext = os.path.splitext(src_path)[1].lower()
    # ↑ Extrae extensión del archivo y la normaliza a minúsculas.
    if ext not in {".ps", ".eps"}:
        # ↑ Si no es PS/EPS, no hay nada que convertir.
        return []

    created = []
    # ↑ Lista de archivos creados (PDF/JPG) para reportar resultados.
    base_name = os.path.splitext(os.path.basename(src_path))[0]
    # ↑ Nombre base del archivo sin extensión.

    ps2pdf_bin = shutil.which("ps2pdf")
    # ↑ Busca el ejecutable ps2pdf en el PATH.
    magick_bin = shutil.which("magick")
    # ↑ Busca "magick" (ImageMagick moderno).
    convert_bin = shutil.which("convert")
    # ↑ Fallback a "convert" (ImageMagick clásico).

    # 1) PS/EPS -> PDF
    # ↑ Primera conversión: generar PDF.
    pdf_path = os.path.join(out_dir, f"{base_name}.pdf")
    # ↑ Ruta destino del PDF.
    if ps2pdf_bin:
        # ↑ Solo intenta si ps2pdf está instalado.
        try:
            subprocess.run([ps2pdf_bin, src_path, pdf_path], check=True)
            # ↑ Ejecuta conversión PS/EPS -> PDF.
            if os.path.exists(pdf_path):
                # ↑ Verifica que realmente se creó el archivo.
                created.append(pdf_path)
                # ↑ Registra el PDF generado.
        except Exception as e:
            # ↑ Captura errores de conversión para no detener todo el workflow.
            print(f"  [diag] Error convirtiendo a PDF ({src_path}): {e}")
    else:
        # ↑ Si no hay ps2pdf, se informa pero no se falla.
        print("  [diag] 'ps2pdf' no está instalado; no se pudo generar PDF.")

    # 2) PS/EPS -> JPG (vía ImageMagick)
    # ↑ Segunda conversión: generar JPG para visualización rápida.
    jpg_path = os.path.join(out_dir, f"{base_name}.jpg")
    # ↑ Ruta destino del JPG.
    if magick_bin or convert_bin:
        # ↑ Solo entra si existe alguna variante de ImageMagick.
        if magick_bin:
            # ↑ Si existe "magick", se usa ese comando (recomendado).
            cmd = [magick_bin, "-density", "300", src_path, "-quality", "95", jpg_path]
            # ↑ -density mejora resolución de rasterización; -quality para compresión JPG.
        else:
            # ↑ Si no existe "magick", intenta con "convert".
            cmd = [convert_bin, "-density", "300", src_path, "-quality", "95", jpg_path]

        try:
            subprocess.run(cmd, check=True)
            # ↑ Ejecuta la conversión a JPG.
            if os.path.exists(jpg_path):
                # ↑ Verifica creación.
                created.append(jpg_path)
                # ↑ Registra el JPG.
        except Exception as e:
            # ↑ Manejo de errores común (muy típico: policy.xml bloquea PS/EPS).
            print(f"  [diag] Error convirtiendo a JPG ({src_path}): {e}")
            print("  [diag] Si ves 'not authorized', hay que habilitar PS/EPS en policy.xml de ImageMagick.")
    else:
        # ↑ Si no hay ImageMagick, se informa.
        print("  [diag] 'magick'/'convert' no está instalado; no se pudo generar JPG.")

    return created
    # ↑ Devuelve lista de rutas creadas (puede estar vacía).
