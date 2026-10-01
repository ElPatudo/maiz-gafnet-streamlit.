from __future__ import annotations

import io
import tempfile
import zipfile
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

from gafnet_model import DEFAULT_WEIGHTS_URL, load_gafnet_model, prepare_checkpoint

st.set_page_config(
    page_title="Cuidado del Maíz - GAF-Net",
    page_icon="🌽",
    layout="wide",
    initial_sidebar_state="expanded",
)

SPANISH = {
    "blight": "Tizón foliar",
    "common_rust": "Roya común",
    "gray_spot": "Mancha gris",
    "health": "Hoja aparentemente sana",
}

CARE_NOTES = {
    "blight": "Revise lesiones alargadas o necrosis. Confirme en campo con apoyo agronómico antes de decidir manejo.",
    "common_rust": "Revise pústulas rojizas o anaranjadas. Compare varias hojas de la planta y registre el lote.",
    "gray_spot": "Revise manchas rectangulares grisáceas entre nervaduras. Tome más fotografías si la lesión es pequeña.",
    "health": "La hoja no mostró señales detectadas por el modelo por encima del umbral usado.",
}


@st.cache_resource(show_spinner=False)
def get_model():
    return load_gafnet_model()


@st.cache_data(show_spinner=False)
def get_checkpoint_path_message() -> str:
    path = prepare_checkpoint()
    size_mb = path.stat().st_size / (1024**2)
    return f"Checkpoint: `{path}` ({size_mb:.2f} MB)"


def image_to_bytes(image: Image.Image, fmt: str = "JPEG") -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format=fmt, quality=95)
    return buffer.getvalue()


def dataframe_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")


def run_prediction(uploaded_file, model, confidence: float, iou: float, img_size: int):
    suffix = Path(uploaded_file.name).suffix or ".jpg"
    raw_bytes = uploaded_file.getvalue()

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(raw_bytes)
        tmp_path = tmp.name

    try:
        results = model.predict(
            source=tmp_path,
            conf=confidence,
            iou=iou,
            imgsz=img_size,
            verbose=False,
            save=False,
        )
    finally:
        try:
            Path(tmp_path).unlink(missing_ok=True)
        except Exception:
            pass

    result = results[0]
    annotated_bgr = result.plot()
    annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
    annotated_image = Image.fromarray(annotated_rgb)

    rows = []
    boxes = result.boxes

    if boxes is not None and len(boxes) > 0:
        for detection_id, box in enumerate(boxes, start=1):
            class_id = int(box.cls[0].item())
            confidence_value = float(box.conf[0].item())
            label = str(result.names[class_id]).lower()
            xyxy = box.xyxy[0].detach().cpu().numpy().tolist()
            rows.append(
                {
                    "archivo": uploaded_file.name,
                    "deteccion": detection_id,
                    "clase_modelo": label,
                    "diagnostico": SPANISH.get(label, label),
                    "confianza": confidence_value,
                    "x1": round(xyxy[0], 1),
                    "y1": round(xyxy[1], 1),
                    "x2": round(xyxy[2], 1),
                    "y2": round(xyxy[3], 1),
                    "umbral_confianza": confidence,
                    "iou": iou,
                    "imgsz": img_size,
                }
            )

    detail_df = pd.DataFrame(rows)

    if detail_df.empty:
        summary = {
            "archivo": uploaded_file.name,
            "diagnostico_principal": "Sin detecciones",
            "clase_modelo_principal": "sin_detecciones",
            "confianza_principal": np.nan,
            "numero_detecciones": 0,
            "imagen_resultado": f"detectado_{Path(uploaded_file.name).stem}.jpg",
        }
    else:
        detail_df = detail_df.sort_values("confianza", ascending=False).reset_index(drop=True)
        disease_rows = detail_df[detail_df["clase_modelo"] != "health"]
        principal = disease_rows.iloc[0] if not disease_rows.empty else detail_df.iloc[0]
        summary = {
            "archivo": uploaded_file.name,
            "diagnostico_principal": principal["diagnostico"],
            "clase_modelo_principal": principal["clase_modelo"],
            "confianza_principal": principal["confianza"],
            "numero_detecciones": len(detail_df),
            "imagen_resultado": f"detectado_{Path(uploaded_file.name).stem}.jpg",
        }

    return annotated_image, detail_df, summary


def make_results_zip(annotated_images: dict[str, Image.Image], summary_df: pd.DataFrame, details_df: pd.DataFrame) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("resumen_diagnostico.csv", dataframe_csv_bytes(summary_df))
        zf.writestr("detecciones_detalladas.csv", dataframe_csv_bytes(details_df))
        for filename, image in annotated_images.items():
            zf.writestr(filename, image_to_bytes(image, fmt="JPEG"))
    return buffer.getvalue()


st.title("🌽 Cuidado del maíz con GAF-Net")
st.caption("Detección asistida de enfermedades foliares en imágenes de maíz con YOLOv8 + GSConv + FASFF.")

with st.sidebar:
    st.header("Parámetros de inferencia")
    confidence = st.slider("Umbral de confianza", 0.05, 0.95, 0.25, 0.05)
    iou_threshold = st.slider("Umbral IoU para NMS", 0.10, 0.95, 0.70, 0.05)
    img_size = st.select_slider("Tamaño de inferencia", options=[416, 512, 640, 768, 960, 1280], value=640)

    st.divider()
    st.markdown("**Clases del modelo**")
    st.markdown("- Tizón foliar (`blight`)\n- Roya común (`common_rust`)\n- Mancha gris (`gray_spot`)\n- Hoja aparentemente sana (`health`)")

    with st.expander("Fuente de pesos"):
        st.code(DEFAULT_WEIGHTS_URL, language="text")
        st.write("La app descarga el checkpoint automáticamente y lo guarda en caché.")

st.info(
    "Esta herramienta es experimental y no reemplaza la evaluación de un agrónomo. "
    "La precisión puede cambiar con fotografías de campo, iluminación, variedad, etapa fenológica y dominio de captura."
)

uploaded_files = st.file_uploader(
    "Carga una o varias fotografías de hojas de maíz",
    type=["jpg", "jpeg", "png", "bmp", "webp"],
    accept_multiple_files=True,
)

col_load, col_status = st.columns([1, 2])
with col_load:
    analyze = st.button("Analizar fotografías", type="primary", disabled=not uploaded_files)
with col_status:
    if uploaded_files:
        st.write(f"Fotografías cargadas: **{len(uploaded_files)}**")

if analyze and uploaded_files:
    with st.spinner("Preparando checkpoint y cargando GAF-Net..."):
        model = get_model()
        checkpoint_message = get_checkpoint_path_message()

    st.success(f"Modelo cargado. {checkpoint_message}")

    summaries = []
    detail_frames = []
    annotated_images = {}

    progress = st.progress(0)
    for idx, uploaded_file in enumerate(uploaded_files, start=1):
        with st.spinner(f"Analizando {uploaded_file.name}..."):
            annotated_image, detail_df, summary = run_prediction(
                uploaded_file,
                model,
                confidence,
                iou_threshold,
                img_size,
            )

        summaries.append(summary)
        if not detail_df.empty:
            detail_frames.append(detail_df)
        annotated_name = summary["imagen_resultado"]
        annotated_images[annotated_name] = annotated_image

        st.subheader(f"Resultado: {uploaded_file.name}")
        left, right = st.columns([1.2, 1])
        with left:
            st.image(annotated_image, caption="Imagen marcada por el modelo", use_container_width=True)
        with right:
            diagnostic = summary["diagnostico_principal"]
            conf = summary["confianza_principal"]
            if pd.isna(conf):
                st.metric("Diagnóstico principal", diagnostic)
            else:
                st.metric("Diagnóstico principal", diagnostic, f"{conf * 100:.2f}%")

            class_key = summary.get("clase_modelo_principal", "")
            if class_key in CARE_NOTES:
                st.write(CARE_NOTES[class_key])

            st.write(f"Número de detecciones: **{summary['numero_detecciones']}**")
            if detail_df.empty:
                st.warning("No hubo detecciones por encima del umbral seleccionado.")
            else:
                display_df = detail_df.copy()
                display_df["confianza"] = (display_df["confianza"] * 100).round(2).astype(str) + "%"
                st.dataframe(display_df, use_container_width=True, hide_index=True)

        progress.progress(idx / len(uploaded_files))

    summary_df = pd.DataFrame(summaries)
    details_df = pd.concat(detail_frames, ignore_index=True) if detail_frames else pd.DataFrame(
        columns=[
            "archivo",
            "deteccion",
            "clase_modelo",
            "diagnostico",
            "confianza",
            "x1",
            "y1",
            "x2",
            "y2",
            "umbral_confianza",
            "iou",
            "imgsz",
        ]
    )

    st.divider()
    st.header("Resumen general")
    visual_summary = summary_df.copy()
    if "confianza_principal" in visual_summary.columns:
        visual_summary["confianza_principal"] = (visual_summary["confianza_principal"] * 100).round(2)
    st.dataframe(visual_summary, use_container_width=True, hide_index=True)

    zip_bytes = make_results_zip(annotated_images, summary_df, details_df)

    d1, d2, d3 = st.columns(3)
    with d1:
        st.download_button(
            "Descargar resumen CSV",
            data=dataframe_csv_bytes(summary_df),
            file_name="resumen_diagnostico.csv",
            mime="text/csv",
        )
    with d2:
        st.download_button(
            "Descargar detecciones CSV",
            data=dataframe_csv_bytes(details_df),
            file_name="detecciones_detalladas.csv",
            mime="text/csv",
        )
    with d3:
        st.download_button(
            "Descargar ZIP completo",
            data=zip_bytes,
            file_name="resultados_GAFNet_maiz.zip",
            mime="application/zip",
        )

else:
    st.markdown(
        "### Flujo de uso\n"
        "1. Ajusta el umbral de confianza en la barra lateral.\n"
        "2. Carga una o varias imágenes de hojas de maíz.\n"
        "3. Presiona **Analizar fotografías**.\n"
        "4. Revisa cajas, diagnóstico principal, confianza y tablas descargables."
    )
