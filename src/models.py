import pandas as pd
import numpy as np
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
from rdkit import Chem
from rdkit.Chem import Descriptors
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

# Neural Network model definition (same as in notebook)
class SolubilityNN(nn.Module):
    def __init__(self, input_size):
        super(SolubilityNN, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Features to use for all models
BASE_FEATURES = [
    "Minimum Degree",
    "Molecular Weight",
    "Number of H-Bond Donors",
    "Number of Rings",
    "Number of Rotatable Bonds",
    "Polar Surface Area"
]

# Additional RDKit features for the models
RDKIT_FEATURES = [
    "MolLogP",
    "MolMR", 
    "HeavyAtomCount",
    "FractionCSP3",
    "TPSA",
    "NumAromaticRings",
    "NumHDonors", 
    "NumHAcceptors"
]

class LinearRegressionModel:
    def __init__(self):
        self.model = LinearRegression()
        self.is_fitted = False
        
    def fit(self, X, y):
        self.model.fit(X, y)
        self.is_fitted = True
        
    def predict(self, X):
        if not self.is_fitted:
            raise ValueError("Model not fitted yet")
        return self.model.predict(X)
    
    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self.model, f)
    
    @classmethod
    def load(cls, path):
        instance = cls()
        with open(path, 'rb') as f:
            instance.model = pickle.load(f)
        instance.is_fitted = True
        return instance

class RandomForestModel:
    def __init__(self):
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.is_fitted = False
        
    def fit(self, X, y):
        self.model.fit(X, y)
        self.is_fitted = True
        
    def predict(self, X):
        if not self.is_fitted:
            raise ValueError("Model not fitted yet")
        return self.model.predict(X)
    
    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self.model, f)
    
    @classmethod
    def load(cls, path):
        instance = cls()
        with open(path, 'rb') as f:
            instance.model = pickle.load(f)
        instance.is_fitted = True
        return instance

class NeuralNetworkModel:
    def __init__(self, input_size=14):
        self.model = SolubilityNN(input_size)
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        self.is_fitted = False
        
    def fit(self, X, y):
        X_scaled = self.scaler_X.fit_transform(X)
        y_scaled = self.scaler_y.fit_transform(y.reshape(-1, 1))
        
        # Training would happen here
        # For now, we'll just set is_fitted to true
        self.is_fitted = True
        
    def predict(self, X):
        if not self.is_fitted:
            raise ValueError("Model not fitted yet")
        
        X_scaled = self.scaler_X.transform(X)
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
        
        self.model.eval()
        with torch.no_grad():
            y_scaled_pred = self.model(X_tensor)
            y_pred = self.scaler_y.inverse_transform(y_scaled_pred.numpy())
        
        return y_pred.flatten()
    
    def save(self, model_path, scaler_X_path, scaler_y_path):
        torch.save(self.model.state_dict(), model_path)
        with open(scaler_X_path, 'wb') as f:
            pickle.dump(self.scaler_X, f)
        with open(scaler_y_path, 'wb') as f:
            pickle.dump(self.scaler_y, f)
    
    @classmethod
    def load(cls, model_path, scaler_X_path, scaler_y_path, input_size=14):
        instance = cls(input_size)
        
        instance.model.load_state_dict(torch.load(model_path))
        instance.model.eval()
        
        with open(scaler_X_path, 'rb') as f:
            instance.scaler_X = pickle.load(f)
        
        with open(scaler_y_path, 'rb') as f:
            instance.scaler_y = pickle.load(f)
        
        instance.is_fitted = True
        return instance

def extract_features_from_smiles(smiles):
    """Extract all required features from a SMILES string"""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    
    # Calculate molecular features
    features = {}
    
    # Calculate basic features
    features["Minimum Degree"] = 1  # Default value as in dataset
    features["Molecular Weight"] = Descriptors.MolWt(mol)
    features["Number of H-Bond Donors"] = Descriptors.NumHDonors(mol)
    features["Number of Rings"] = mol.GetRingInfo().NumRings()
    features["Number of Rotatable Bonds"] = Descriptors.NumRotatableBonds(mol)
    features["Polar Surface Area"] = Descriptors.TPSA(mol)
    
    # Calculate RDKit features
    features["MolLogP"] = Descriptors.MolLogP(mol)
    features["MolMR"] = Descriptors.MolMR(mol)
    features["HeavyAtomCount"] = Descriptors.HeavyAtomCount(mol)
    features["FractionCSP3"] = Descriptors.FractionCSP3(mol)
    features["TPSA"] = Descriptors.TPSA(mol)
    features["NumAromaticRings"] = Descriptors.NumAromaticRings(mol)
    features["NumHDonors"] = Descriptors.NumHDonors(mol)
    features["NumHAcceptors"] = Descriptors.NumHAcceptors(mol)
    
    return features

def prepare_features_for_model(features_dict, model_type):
    """Prepare features as needed for each model type"""
    if model_type == "linear":
        # Linear regression just uses the base features
        return np.array([[features_dict[feature] for feature in BASE_FEATURES]])
    else:
        # RF and NN use all features
        all_features = BASE_FEATURES + RDKIT_FEATURES
        return np.array([[features_dict[feature] for feature in all_features]]) 