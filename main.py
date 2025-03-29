from fastapi import FastAPI
import pickle
import pandas as pd
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# Cargar los modelos
with open('RFCropRecommendation.pkl', 'rb') as f:
    rf_model = pickle.load(f)

with open('SVMModel.pkl', 'rb') as f:
    svm_model = pickle.load(f)

# Crear aplicación FastAPI
app = FastAPI()

# Configuración de CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Permitir acceso desde cualquier origen
    allow_credentials=True,
    allow_methods=["*"],  # Permitir todos los métodos HTTP (POST, GET, OPTIONS, etc.)
    allow_headers=["*"],  # Permitir todos los encabezados HTTP
)

# Definir el esquema de entrada usando Pydantic
class CropInput(BaseModel):
    N: float
    P: float
    K: float
    temperature: float
    humidity: float
    ph: float
    rainfall: float

# Ruta para manejar las solicitudes OPTIONS manualmente
@app.options("/predict-rf/")
async def handle_options():
    return JSONResponse(status_code=200, content={"message": "CORS preflight successful"})

# Definir el endpoint para el modelo RandomForest
@app.post("/predict-rf/")
def predict_rf(input_data: CropInput):
    input_dict = input_data.dict()
    input_df = pd.DataFrame([input_dict])
    prediction = rf_model.predict(input_df)
    return {"prediction": prediction[0]}

# Definir el endpoint para el modelo SVM
@app.post("/predict-svm/")
def predict_svm(input_data: CropInput):
    input_dict = input_data.dict()
    input_df = pd.DataFrame([input_dict])
    prediction = svm_model.predict(input_df)
    return {"prediction": prediction[0]}

# Ruta de bienvenida
@app.get("/")
def read_root():
    return {"message": "API para la recomendación de cultivos mediante múltiples modelos"}
