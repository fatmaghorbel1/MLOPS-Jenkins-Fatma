from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
from  typing import List
# Define the RetrainData class BEFORE using it
class RetrainData(BaseModel):
    hyperparameters: dict  # Dictionary of XGBoost hyperparameters
    new_data: List[List[float]] = None  # New training data (optional)
    new_labels: List[float] = None      # New labels (optional)
# Charger le modèle préalablement sauvegardé
model = joblib.load("model.pkl")

# Créer une instance de l'application FastAPI
app = FastAPI()

# Définir le schéma de données attendu pour la requête
class InputData(BaseModel):
    account_length: int
    area_code: int
    international_plan: str
    voice_mail_plan: str
    number_vmail_messages: int
    total_day_minutes: float
    total_day_calls: int
    total_day_charge: float
    total_eve_minutes: float
    total_eve_calls: int
    total_eve_charge: float
    total_night_minutes: float
    total_night_calls: int
    total_night_charge: float
    total_intl_minutes: float
    total_intl_calls: int
    total_intl_charge: float
    customer_service_calls: int
    churn: bool  # La variable 'churn' est incluse mais elle ne devrait pas l'être pour la prédiction

# Définir la route HTTP POST pour effectuer la prédiction
@app.post("/predict/")
async def predict(data: InputData):
    try:
        # Convertir les variables catégorielles en valeurs numériques (1 pour 'yes', 0 pour 'no')
        international_plan = 1 if data.international_plan == 'yes' else 0
        voice_mail_plan = 1 if data.voice_mail_plan == 'yes' else 0

        # Préparer les données d'entrée pour la prédiction
        input_data = np.array([[
            data.account_length,
            data.area_code,
            international_plan,  # valeurs numériques
            voice_mail_plan,     # valeurs numériques
            data.number_vmail_messages,
            data.total_day_minutes,
            data.total_day_calls,
            data.total_day_charge,
            data.total_eve_minutes,
            data.total_eve_calls,
            data.total_eve_charge,
            data.total_night_minutes,
            data.total_night_calls,
            data.total_night_charge,
            data.total_intl_minutes,
            data.total_intl_calls,
            data.total_intl_charge,
            data.customer_service_calls,
            data.churn
        ]])

        # Debugging: Afficher les données d'entrée
        print(f"Input data: {input_data}")

        # Effectuer la prédiction avec le modèle
        prediction = model.predict(input_data)

        # Retourner le résultat de la prédiction
        return {"prediction": prediction.tolist()}

    except Exception as e:
        # Log l'erreur pour mieux comprendre la source du problème
        print(f"Error during prediction: {e}")
        raise HTTPException(status_code=500, detail=f"Error during prediction: {e}")

 
# Définir la route HTTP POST pour réentraîner le modèle
@app.post("/retrain/")
async def retrain(data: RetrainData):
    try:
        global model
        model = joblib.load("model.pkl")

        # Update XGBoost hyperparameters
        for param, value in data.hyperparameters.items():
            if hasattr(model, param):
                setattr(model, param, value)
            else:
                raise HTTPException(status_code=400, detail=f"Invalid hyperparameter: {param}")

        # Retrain the model with new data (if provided)
        if data.new_data and data.new_labels:
            X_train = np.array(data.new_data)
            y_train = np.array(data.new_labels)
            model.fit(X_train, y_train)
        else:
            # Retrain with existing data (you need to load this)
            # X_train, y_train = load_existing_data()
            # model.fit(X_train, y_train)
            pass

        # Save the updated model
        joblib.dump(model, "model.pkl")

        return {"message": "Model retrained successfully"}

    except Exception as e:
        print(f"Error during retraining: {e}")
        raise HTTPException(status_code=500, detail=f"Error during retraining: {e}")



