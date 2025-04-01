import collections
import collections.abc

# Restaurar la compatibilidad con versiones anteriores de collections
for type_name in ('Iterable', 'Mapping', 'MutableSet', 'Sequence'):
    if not hasattr(collections, type_name) and hasattr(collections.abc, type_name):
        setattr(collections, type_name, getattr(collections.abc, type_name))

import os
import requests
import base64
import io
import numpy as np
from PIL import Image
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS

# URLs de las APIs
CLASSIFIER_API_URL = "https://apiclassifier.onrender.com/predict"
SEGMENTATION_API_URL = "https://tumor-segmentation-api-latest.onrender.com/segment"

app = Flask(__name__, template_folder='templates')
CORS(app)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        file = request.files['image']
        
        # Guardar datos de la imagen para codificar
        file_data = file.read()
        file.seek(0)  # Reiniciar puntero del archivo
        
        # Crear URI de datos para la imagen original
        mime_type = file.content_type if hasattr(file, 'content_type') else 'image/jpeg'
        encoded_img = f"data:{mime_type};base64,{base64.b64encode(file_data).decode('utf-8')}"

        # Enviar imagen al modelo de clasificación
        files = {'image': file}
        classifier_response = requests.post(CLASSIFIER_API_URL, files=files)
        classifier_data = classifier_response.json()

        if 'error' in classifier_data:
            raise Exception(f"Error en la clasificación: {classifier_data['error']}")

        tumor_prob = classifier_data['tumor_probability']

        # Solo continuar si la probabilidad de tumor es mayor al 50%
        if tumor_prob <= 0.5:
            return render_template('index.html', result="No hay tumor", 
                                  probability=float(tumor_prob), 
                                  original_image=encoded_img)

        # Enviar imagen al modelo de segmentación
        file.seek(0)
        segmentation_response = requests.post(SEGMENTATION_API_URL, files=files)
        segmentation_data = segmentation_response.json()

        if 'error' in segmentation_data:
            raise Exception(f"Error en la segmentación: {segmentation_data['error']}")

        # Procesar la máscara de segmentación
        segmentation_mask = np.array(segmentation_data['segmentation_mask'], dtype=np.uint8) * 255
        
        # Convertir máscara a imagen y codificar
        mask_img = Image.fromarray(segmentation_mask)
        mask_buffer = io.BytesIO()
        mask_img.save(mask_buffer, format='PNG')
        encoded_mask = f"data:image/png;base64,{base64.b64encode(mask_buffer.getvalue()).decode('utf-8')}"
        
        # Obtener imagen superpuesta si está disponible
        overlay_image = None
        if 'overlay_image' in segmentation_data:
            overlay_image = f"data:image/png;base64,{segmentation_data['overlay_image']}"

        return render_template('index.html', result="Tumor detectado", 
                              probability=float(tumor_prob), 
                              mask=encoded_mask,
                              overlay_image=overlay_image,
                              original_image=encoded_img)

    except Exception as e:
        return render_template('index.html', error=str(e))

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5005))
    app.run(host="0.0.0.0", port=port)
