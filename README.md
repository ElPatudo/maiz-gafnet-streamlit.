# Cuidado del Maíz - Aplicación Streamlit con GAF-Net

Aplicación web para detección asistida de enfermedades en hojas de maíz usando el checkpoint público de **GAF-Net Multi-Disease Corn Leaf Detection**, basado en YOLOv8 con módulos personalizados **GSConv** y **FASFF**.

## Clases del modelo

| Clase del checkpoint | Nombre mostrado en la app |
|---|---|
| `blight` | Tizón foliar |
| `common_rust` | Roya común |
| `gray_spot` | Mancha gris |
| `health` | Hoja aparentemente sana |

## Estructura del proyecto

```text
maiz_gafnet_streamlit/
├── streamlit_app.py              # Aplicación principal
├── gafnet_model.py                # Registro de GSConv/FASFF, descarga y carga del modelo
├── requirements.txt               # Dependencias para Streamlit Cloud/GitHub
├── .streamlit/config.toml         # Tema visual y configuración básica
├── models/README.md               # Opción manual para pesos locales
├── original/                      # Notebook Colab original de referencia
├── docs/                          # Tutorial Beamer en PDF y .tex
├── run_local_windows.bat          # Ejecución local en Windows
├── run_local_linux_mac.sh         # Ejecución local en Linux/macOS
└── .gitignore                     # Evita subir cachés y pesos pesados
```

## Uso recomendado

No subas `best.pt` al repositorio. La aplicación descarga automáticamente `best.pt.zip` desde el repositorio público original y lo guarda en caché como `model_cache/best.pt`.

Si la descarga automática falla, puedes usar la ruta alternativa:

1. Descarga manualmente `best.pt.zip` desde el repositorio original.
2. Coloca el checkpoint como `models/best.pt`.
3. Ejecuta nuevamente la aplicación.

## Ejecución local rápida

### Windows

```bat
run_local_windows.bat
```

### Linux/macOS

```bash
bash run_local_linux_mac.sh
```

### Manual

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run streamlit_app.py
```

## Despliegue en Streamlit Community Cloud

1. Crea un repositorio nuevo en GitHub.
2. Sube todos los archivos de esta carpeta.
3. En Streamlit Community Cloud, selecciona el repositorio.
4. Define como archivo principal: `streamlit_app.py`.
5. En configuración avanzada, usa Python 3.11 si está disponible.
6. Despliega la app.

## Archivos de salida de la aplicación

La aplicación permite descargar:

- `resumen_diagnostico.csv`: diagnóstico principal por imagen.
- `detecciones_detalladas.csv`: clase, confianza y coordenadas por caja.
- `resultados_GAFNet_maiz.zip`: imágenes marcadas + CSV.

## Nota importante

Esta herramienta es experimental. No reemplaza un diagnóstico agronómico profesional. Las métricas del proyecto original corresponden al conjunto de evaluación usado por los autores y no garantizan el mismo desempeño en fotografías tomadas en otros cultivos, cámaras, iluminaciones o condiciones de campo.
