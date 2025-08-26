Clasificador Biomédico Multietiqueta (Cardio / Neuro / Hepatorrenal / Onco)

Demo (Hugging Face): https://huggingface.co/spaces/jennifersalazarduke/clasificador-biomedicoTECH_SPHERE

Storytelling (Notion): https://humorous-polyester-33a.notion.site/Challenge-de-Clasificaci-n-Biom-dica-con-IA-25a30d9b1a1180dba80de926852cc7fd?pvs=74

Repositorio público (GitHub): https://github.com/SalazarDukeImpactHub/tech-sphere-clasificador-biomedico-multietiqueta

TL;DR: Este proyecto clasifica artículos médicos (usando título + abstract) en una o varias áreas: Cardiovascular, Neurológico, Hepatorrenal, Oncológico. El modelo principal usa TF-IDF (palabra+carácter) + Regresión Logística OVR y alcanza F1 ponderado ≈ 0.905. Incluye: cuaderno de entrenamiento, artefactos del modelo listos para producción, API REST y demo web. Además, se aporta un dashboard V0 para comunicar resultados y enlaces a un storytelling en Notion.

🌟 Qué hace y para quién es

Qué hace: Lee title+abstract, calcula probabilidades por tema y asigna múltiples etiquetas cuando corresponde.

Para quién: Perfiles junior/intermedios en datos/IA, equipos clínicos o de producto que requieran priorizar lectura de literatura biomédica.

🚀 Resultados (hold-out 20%)

F1 ponderado ≈ 0.905

Métricas por tema (aprox.):

Cardiovascular 0.92 · Neurológico 0.90 · Hepatorrenal 0.90 · Oncológico 0.88

Umbrales por clase (producción): [0.43, 0.58, 0.48, 0.50]

Nota: el sistema nunca devuelve vacío: si ninguna prob supera su umbral, usa Top-1 como respaldo.

🧠 Enfoque del modelo (breve y entendible)

Texto → números: TF-IDF por palabra (1–2) y por carácter (3–5) para captar términos y abreviaturas.

Aprendizaje: Regresión Logística en esquema One-Vs-Rest (una por tema).

Decisión: umbrales por clase afinados para maximizar F1 ponderado.

Multietiqueta: puede activar varias áreas si el artículo es mixto.

🗂️ Estructura del repositorio
tech-sphere-clasificador-biomedico-multietiqueta/
├─ data/                       # dataset (ej. challenge_data-18-ago.csv)*
├─ models_final/               # artefactos de producción
│  ├─ pipeline.joblib          # vectorizadores + clasificador
│  └─ meta.json                # labels, thresholds, engine
├─ notebooks/
│  └─ intento1.ipynb           # cuaderno final organizado (entrenamiento/validación)
├─ src/
│  ├─ app.py                   # API REST (FastAPI)
│  └─ streamlit_app.py         # demo local/Space
├─ v0/                         # prompts y recursos de visualización (V0)
├─ reports_env/                # environment.yml, requirements.txt, reportes de entorno
├─ README.md
└─ .gitignore


* Si el dataset es sensible/privado, no subir a GitHub (.gitignore).

🔗 Enlaces importantes

Demo en vivo (Hugging Face Space): abre el formulario, pega título+abstract y visualiza etiquetas y barras de confianza.
https://huggingface.co/spaces/jennifersalazarduke/clasificador-biomedicoTECH_SPHERE

Storytelling del proyecto (Notion): relato amigable del reto, decisiones y resultados.
https://humorous-polyester-33a.notion.site/Challenge-de-Clasificaci-n-Biom-dica-con-IA-25a30d9b1a1180dba80de926852cc7fd?pvs=74

Repositorio público (GitHub): código fuente, artefactos y documentación.
https://github.com/SalazarDukeImpactHub/tech-sphere-clasificador-biomedico-multietiqueta

🧪 Reproducir localmente (entorno biomed-ml)

Recomendado: Python 3.10. Estos pasos fueron probados en Anaconda Prompt (Windows).

# 1) Crear/activar entorno
conda create -n biomed-ml python=3.10 -y
conda activate biomed-ml
python -m pip install --upgrade pip

# 2) Instalar dependencias
python -m pip install -r requirements.txt
# (si no existe, instalar mínimas)
# python -m pip install scikit-learn==1.7.1 numpy pandas joblib fastapi uvicorn streamlit

# 3) Verificar versiones clave (opcional)
python -c "import sklearn,streamlit,sys; print('sklearn', sklearn.__version__, '| streamlit', streamlit.__version__, '| py', sys.version)"


Entrenamiento y validación
Abre notebooks/intento1.ipynb, selecciona el kernel Python (biomed-ml) y ejecuta.
Al finalizar, se generan models_final/pipeline.joblib y models_final/meta.json.

🔌 API REST (FastAPI)

Arranca la API que sirve las predicciones (JSON):

# en la raíz del repo
set MODEL_DIR=models_final
uvicorn src.app:app --host 0.0.0.0 --port 8000


Endpoints

GET /health → estado, labels, thresholds

POST /predict → body:

{ "title": "string", "abstract": "string", "top_k": 0, "fallback": true }


Modo umbrales (top_k=0) o Top-K (top_k>0).

▶️ Demo web (Streamlit)

Ejecuta localmente la demo (la misma que se usa en el Space):

streamlit run src/streamlit_app.py

📊 Visualización con V0 (bonus)

En v0/ encontrarás prompts para crear un dashboard de una sola página (KPIs, métricas por tema con barras de colores diferenciados, y panel de errores por tema).
El dashboard incluye botones a:

Probar la demo (Hugging Face)

Storytelling (Notion)

Código (GitHub)

Pensado para comunicar resultados a público no técnico, con tooltips claros y sin jerga.

✅ Checklist de entrega

 Cuaderno final organizado (notebooks/intento1.ipynb)

 Modelo listo para servir (models_final/)

 API REST (src/app.py)

 Demo web (src/streamlit_app.py) y Space público

 V0 dashboard (prompts y recursos)

 Reportes de entorno (reports_env/) para reproducibilidad

🔒 Consideraciones y límites

No es un dispositivo médico; requiere revisión humana.

Textos muy cortos pueden ser ambiguos.

Para mayor recall en clases minoritarias, considerar BioBERT/ClinicalBERT (pipeline alterno).

👩‍💻 Autoría

Jennifer Salazar Duke — Salazar Duke Impact Hub

¿Preguntas o mejoras? Abre un Issue o un Pull Request en este repositorio.
