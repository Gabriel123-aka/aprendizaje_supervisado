from flask import Flask, request, jsonify
import pickle
import pandas as pd

app = Flask(__name__)

# 1. Cargamos el PIPELINE completo (no solo el modelo solo)
# Asegúrate de que el archivo pipeline.pkl esté en la misma carpeta que este script
with open('pipeline.pkl', 'rb') as archivo_modelo:
    modelo = pickle.load(archivo_modelo)

@app.route('/predecir', methods=['POST'])
def predecir():
    # 2. Obtenemos los datos crudos del JSON
    data = request.get_json()

    # 3. Convertimos a DataFrame (esto permite que el Pipeline procese "male", "C", etc.)
    input_data = pd.DataFrame([data])

    # 4. Hacemos la predicción (el pipeline limpia y escala los datos automáticamente)
    prediccion = modelo.predict(input_data)
    
    # 5. Devolvemos la respuesta
    output = {'Survived': int(prediccion[0])}
    
    return jsonify(output)

if __name__ == '__main__':
    # Ejecutamos el servidor
    app.run(debug=True)