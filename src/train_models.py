import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from rdkit import Chem
from rdkit.Chem import Descriptors

from models import (
    LinearRegressionModel, 
    RandomForestModel, 
    NeuralNetworkModel,
    BASE_FEATURES,
    RDKIT_FEATURES
)

# Create models directory if it doesn't exist
os.makedirs('models', exist_ok=True)

# Load and preprocess data
print("Loading data...")
df = pd.read_csv("data/delaney-processed.csv")

# Add RDKit features
print("Calculating additional features...")
df["rdkit_mol"] = df["smiles"].apply(Chem.MolFromSmiles)
df["MolLogP"] = df["rdkit_mol"].apply(Descriptors.MolLogP)
df["MolMR"] = df["rdkit_mol"].apply(Descriptors.MolMR)
df["HeavyAtomCount"] = df["rdkit_mol"].apply(Descriptors.HeavyAtomCount)
df["FractionCSP3"] = df["rdkit_mol"].apply(Descriptors.FractionCSP3)
df["TPSA"] = df["rdkit_mol"].apply(Descriptors.TPSA)
df["NumAromaticRings"] = df["rdkit_mol"].apply(Descriptors.NumAromaticRings)
df["NumHDonors"] = df["rdkit_mol"].apply(Descriptors.NumHDonors)
df["NumHAcceptors"] = df["rdkit_mol"].apply(Descriptors.NumHAcceptors)

target = "measured log solubility in mols per litre"

# Prepare datasets for each model
X_linear = df[BASE_FEATURES].values
X_full = df[BASE_FEATURES + RDKIT_FEATURES].values
y = df[target].values

# Split data
X_linear_train, X_linear_test, y_train, y_test = train_test_split(
    X_linear, y, test_size=0.2, random_state=42
)
X_full_train, X_full_test, _, _ = train_test_split(
    X_full, y, test_size=0.2, random_state=42
)

# 1. Train and save Linear Regression model
print("Training Linear Regression model...")
lr_model = LinearRegressionModel()
lr_model.fit(X_linear_train, y_train)
lr_model.save('models/linear_regression.pkl')
print(f"Linear Regression R^2: {lr_model.model.score(X_linear_test, y_test):.4f}")

# 2. Train and save Random Forest model
print("Training Random Forest model...")
rf_model = RandomForestModel()
rf_model.fit(X_full_train, y_train)
rf_model.save('models/random_forest.pkl')
rf_preds = rf_model.predict(X_full_test)
rf_r2 = np.corrcoef(y_test, rf_preds)[0, 1]**2
print(f"Random Forest R^2: {rf_r2:.4f}")

# 3. Train and save Neural Network model
print("Training Neural Network model...")
nn_model = NeuralNetworkModel(input_size=X_full.shape[1])

# Prepare data for NN
X_scaled = nn_model.scaler_X.fit_transform(X_full_train)
y_scaled = nn_model.scaler_y.fit_transform(y_train.reshape(-1, 1))

X_train_tensor = torch.tensor(X_scaled, dtype=torch.float32)
y_train_tensor = torch.tensor(y_scaled, dtype=torch.float32)

# Set up dataloader
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Training
nn_model.model.train()
optimizer = optim.Adam(nn_model.model.parameters(), lr=0.0001)
loss_fn = nn.MSELoss()

n_epochs = 100
for epoch in range(n_epochs):
    for xb, yb in train_loader:
        pred = nn_model.model(xb)
        loss = loss_fn(pred, yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    if (epoch + 1) % 20 == 0:
        print(f"Epoch {epoch+1}/{n_epochs}, Loss: {loss.item():.4f}")

# Set is_fitted to True
nn_model.is_fitted = True

# Save the model and scalers
nn_model.save('models/neural_network.pt', 'models/scaler_X.pkl', 'models/scaler_y.pkl')

# Evaluate NN
X_test_scaled = nn_model.scaler_X.transform(X_full_test)
X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)

nn_model.model.eval()
with torch.no_grad():
    y_pred_scaled = nn_model.model(X_test_tensor)
    y_pred = nn_model.scaler_y.inverse_transform(y_pred_scaled.numpy())

nn_r2 = np.corrcoef(y_test, y_pred.flatten())[0, 1]**2
print(f"Neural Network R^2: {nn_r2:.4f}")

print("All models trained and saved successfully!") 