import os
import json
import joblib
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS # Import CORS to allow requests from your frontend

# Inicializa la aplicación Flask
app = Flask(__name__)
CORS(app) # Habilita CORS para todas las rutas, permitiendo que tu frontend se conecte

# --- Carga del Modelo Entrenado ---
# Detecta el directorio del modelo. Asegúrate de que tu carpeta 'models_final' esté en el mismo nivel que app.py
CANDIDATES = ["models_final", "models_opt", "models_py"]
MODEL_DIR = next((d for d in CANDIDATES if os.path.exists(d)), None)

if MODEL_DIR is None:
    raise FileNotFoundError("No se encontró ninguna carpeta de modelo (models_final/models_opt/models_py).")

# Carga el pipeline del modelo
PIPE = joblib.load(os.path.join(MODEL_DIR, "pipeline.joblib"))

# Carga los metadatos (etiquetas y umbrales)
meta_path = os.path.join(MODEL_DIR, "meta.json")
if not os.path.exists(meta_path):
    raise FileNotFoundError(f"No se encontró el archivo meta.json en {MODEL_DIR}")

with open(meta_path, "r", encoding="utf-8") as f:
    META = json.load(f)

LABELS = META.get("labels", ["Cardiovascular", "Neurological", "Hepatorenal", "Oncológica"])
THRESHOLDS = np.array(META.get("thresholds", [0.5]*len(LABELS)), dtype=float)
ENGINE = META.get("engine", "tfidf_pipeline")

print(f"Modelo cargado: {ENGINE} desde {MODEL_DIR}")
print(f"Etiquetas: {LABELS}")
print(f"Umbrales: {THRESHOLDS}")

# --- Endpoint de Predicción ---
@app.route('/predict', methods=['POST'])
def predict():
    """
    Endpoint para realizar predicciones sobre el título y el abstract de un artículo.
    Recibe un JSON con 'title' y 'abstract'.
    Retorna las etiquetas predichas y sus puntuaciones.
    """
    try:
        data = request.get_json()
        title = data.get('title', '')
        abstract = data.get('abstract', '')

        if not title and not abstract:
            return jsonify({'error': 'Se requiere al menos un título o un abstract para la predicción.'}), 400

        text = f"{title or ''} {abstract or ''}".strip()

        # Realiza la predicción de probabilidades
        P = PIPE.predict_proba([text])[0] # Obtiene las probabilidades para la única muestra

        # Determina las etiquetas predichas basándose en los umbrales
        predicted_labels = [
            LABELS[i] for i in range(len(LABELS)) if P[i] >= THRESHOLDS[i]
        ]
        
        # Si ninguna etiqueta supera el umbral, se devuelve la de mayor puntuación (fallback)
        if not predicted_labels and len(LABELS) > 0:
            top_idx = np.argmax(P)
            predicted_labels = [LABELS[top_idx]]

        # Formatea los scores para la respuesta
        scores_sorted = []
        order = np.argsort(-P) # Ordena los índices de mayor a menor probabilidad
        for i in order:
            scores_sorted.append({
                "label": LABELS[i],
                "score": float(P[i]),
                "threshold": float(THRESHOLDS[i]),
                "selected": (LABELS[i] in predicted_labels)
            })

        response = {
            "engine": ENGINE,
            "predicted": predicted_labels,
            "scores_sorted": scores_sorted,
            "model_dir": MODEL_DIR
        }

        return jsonify(response)

    except Exception as e:
        app.logger.error(f"Error en la predicción: {e}")
        return jsonify({'error': f'Ocurrió un error interno en el servidor: {str(e)}'}), 500

# --- Inicio del Servidor ---
if __name__ == '__main__':
    # Asegúrate de que el directorio del modelo exista
    if not os.path.exists(MODEL_DIR):
        print(f"Error: El directorio del modelo '{MODEL_DIR}' no se encontró.")
        print("Por favor, asegúrate de que la carpeta 'models_final' (o 'models_opt'/'models_py') esté en el mismo directorio que este script.")
    else:
        # Ejecuta la aplicación en modo debug para desarrollo. En producción, usa un servidor WSGI como Gunicorn.
        app.run(debug=True, host='0.0.0.0', port=5000)
