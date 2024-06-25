from flask import Flask, request, render_template, jsonify
import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler
import logging

app = Flask(__name__)

# Configurar el registro
logging.basicConfig(level=logging.DEBUG)

# Cargar el modelo entrenado y el escalador
model = joblib.load('model.pkl')
scaler = joblib.load('scaler.pkl')

app.logger.debug('Modelo y escalador cargados correctamente.')

@app.route('/')
def home():
    return render_template('formulario.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Obtener los datos enviados en el request
        Clump_Thickness = int(request.form['Clump Thickness'])
        Uniformity_of_Cell_Size = int(request.form['Uniformity of Cell Size'])
        Uniformity_of_Cell_Shape = int(request.form['Uniformity of Cell Shape'])
        Bare_Nuclei = int(request.form['Bare Nuclei'])
        
        # Escalar los datos de entrada automáticamente
        input_data = pd.DataFrame([[Clump_Thickness, Uniformity_of_Cell_Size, Uniformity_of_Cell_Shape, Bare_Nuclei]], columns=['Clump Thickness', 'Uniformity of Cell Size', 'Uniformity of Cell Shape', 'Bare Nuclei'])
        input_data_scaled = scaler.transform(input_data)
        
        # Realizar predicciones
        prediction = model.predict(input_data_scaled)
        app.logger.debug(f'Predicción: {prediction[0]}')
        
        # Devolver las predicciones como respuesta JSON
        return jsonify({'categoria': prediction[0]})
    except Exception as e:
        app.logger.error(f'Error en la predicción: {str(e)}')
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
