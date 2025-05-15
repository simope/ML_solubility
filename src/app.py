import os
import numpy as np
from fastapi import FastAPI, Request, Form, HTTPException
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import uvicorn
import rdkit
from rdkit import Chem
from rdkit.Chem import AllChem, Draw
from markupsafe import Markup

from src.models import (
    LinearRegressionModel,
    RandomForestModel,
    NeuralNetworkModel,
    extract_features_from_smiles,
    prepare_features_for_model
)
from src.utils import get_3d_molecule_html

# Create the models directory if it doesn't exist yet
os.makedirs('models', exist_ok=True)

# Create the templates directory if it doesn't exist
templates_dir = os.path.join(os.path.dirname(__file__), '..', 'templates')
os.makedirs(templates_dir, exist_ok=True)

# Create static directory for CSS, JS and images
static_dir = os.path.join(os.path.dirname(__file__), '..', 'static')
os.makedirs(static_dir, exist_ok=True)
os.makedirs(os.path.join(static_dir, 'images'), exist_ok=True)

app = FastAPI(title="Molecule Solubility Prediction")
app.mount("/static", StaticFiles(directory=static_dir), name="static")
templates = Jinja2Templates(directory=templates_dir)

# Load models
try:
    linear_model = LinearRegressionModel.load('models/linear_regression.pkl')
    rf_model = RandomForestModel.load('models/random_forest.pkl')
    nn_model = NeuralNetworkModel.load(
        'models/neural_network.pt',
        'models/scaler_X.pkl',
        'models/scaler_y.pkl'
    )
    models_loaded = True
except Exception as e:
    print(f"Error loading models: {e}")
    print("Please run train_models.py first to train and save the models.")
    models_loaded = False

# Pydantic models for API
class SmilesInput(BaseModel):
    smiles: str
    model_type: str

class PredictionOutput(BaseModel):
    smiles: str
    model_type: str
    solubility_log: float  # Log value
    solubility_mol: float  # Actual molar value
    html_3d: str

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Render the home page with the prediction form"""
    return templates.TemplateResponse(
        "index.html", 
        {"request": request, "models_loaded": models_loaded}
    )

@app.post("/predict", response_class=HTMLResponse)
async def predict(request: Request, smiles: str = Form(...), model_type: str = Form(...)):
    """Handle the form submission and make a prediction"""
    if not models_loaded:
        return templates.TemplateResponse(
            "index.html", 
            {
                "request": request, 
                "error": "Models not loaded. Please run train_models.py first.",
                "models_loaded": models_loaded
            }
        )
    
    # Validate SMILES
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return templates.TemplateResponse(
            "index.html", 
            {
                "request": request, 
                "error": "Invalid SMILES string",
                "models_loaded": models_loaded,
                "smiles": smiles
            }
        )
    
    # Generate 3D molecule visualization HTML
    molecule_html = get_3d_molecule_html(smiles)
    
    # Extract features
    features = extract_features_from_smiles(smiles)
    
    # Select model and make prediction
    prediction_log = None
    if model_type == "linear":
        X = prepare_features_for_model(features, "linear")
        prediction_log = linear_model.predict(X)[0]
    elif model_type == "random_forest":
        X = prepare_features_for_model(features, "random_forest")
        prediction_log = rf_model.predict(X)[0]
    elif model_type == "neural_network":
        X = prepare_features_for_model(features, "neural_network")
        prediction_log = nn_model.predict(X)[0]
    else:
        return templates.TemplateResponse(
            "index.html", 
            {
                "request": request, 
                "error": "Invalid model type",
                "models_loaded": models_loaded,
                "smiles": smiles
            }
        )
    
    # Convert from log(mol/liter) to mol/liter
    prediction_mol = 10 ** prediction_log
    
    # Return the prediction result
    return templates.TemplateResponse(
        "index.html", 
        {
            "request": request, 
            "models_loaded": models_loaded,
            "smiles": smiles,
            "model_type": model_type,
            "prediction_log": prediction_log if prediction_log is not None else None,
            "prediction_mol": prediction_mol if prediction_log is not None else None,
            "molecule_html": Markup(molecule_html)  # Mark as safe HTML
        }
    )

@app.post("/api/predict", response_model=PredictionOutput)
async def predict_api(input_data: SmilesInput):
    """API endpoint for prediction"""
    if not models_loaded:
        raise HTTPException(
            status_code=500, 
            detail="Models not loaded. Please run train_models.py first."
        )
    
    # Validate SMILES
    mol = Chem.MolFromSmiles(input_data.smiles)
    if mol is None:
        raise HTTPException(status_code=400, detail="Invalid SMILES string")
    
    # Generate 3D molecule visualization HTML
    molecule_html = get_3d_molecule_html(input_data.smiles)
    
    # Extract features
    features = extract_features_from_smiles(input_data.smiles)
    
    # Select model and make prediction
    prediction_log = None
    if input_data.model_type == "linear":
        X = prepare_features_for_model(features, "linear")
        prediction_log = linear_model.predict(X)[0]
    elif input_data.model_type == "random_forest":
        X = prepare_features_for_model(features, "random_forest")
        prediction_log = rf_model.predict(X)[0]
    elif input_data.model_type == "neural_network":
        X = prepare_features_for_model(features, "neural_network")
        prediction_log = nn_model.predict(X)[0]
    else:
        raise HTTPException(status_code=400, detail="Invalid model type")
    
    # Convert from log(mol/liter) to mol/liter
    prediction_mol = 10 ** prediction_log
    
    return PredictionOutput(
        smiles=input_data.smiles,
        model_type=input_data.model_type,
        solubility_log=float(prediction_log) if prediction_log is not None else 0.0,
        solubility_mol=float(prediction_mol) if prediction_log is not None else 0.0,
        html_3d=molecule_html
    )

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000) 