# Molecule Solubility Prediction Web App

<<<<<<< HEAD
This project provides a web application for predicting the solubility of chemical compounds using three different machine learning models: Linear Regression, Random Forest, and Neural Network.
=======
This is a work in progress project that explores the prediction of molecular solubility using various machine learning models, including linear regression, random forest regression, and neural networks. The dataset used is based on the Delaney solubility dataset, with engineered features for improved model performance.
>>>>>>> 3a234af8ccca3f92e846d7bae934197f7b8db36a

## Overview

The application allows users to:

1. Input a molecule using SMILES notation
2. Choose between three prediction models:
   - Linear Regression
   - Random Forest Regressor
   - Neural Network
3. Visualize the molecule structure
4. Get solubility predictions in log mols per liter

## Dataset

The models are trained on the Delaney solubility dataset, which includes various molecular descriptors and measured solubility values for about 1,100 compounds.

## Prerequisites

- Python 3.8 or higher
- pip for installing requirements

## Installation

1. Clone this repository:
   ```
   git clone <repository-url>
   cd solubility
   ```

2. Create a virtual environment (optional but recommended):
   ```
   python -m venv .venv
   # On Windows
   .\.venv\Scripts\activate
   # On Linux/Mac
   source .venv/bin/activate
   ```

3. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

## Usage

### 1. Train the Models

Before using the app, you need to train the models:

```
python src/train_models.py
```

This script will train all three models and save them to the `models/` directory.

### 2. Start the Web Application

Launch the FastAPI application:

```
python -m uvicorn src.app:app --reload
```

The web interface will be available at http://localhost:8000

### 3. Using the API

The application also provides a REST API for integration with other systems:

```bash
curl -X POST "http://localhost:8000/api/predict" \
  -H "Content-Type: application/json" \
  -d '{"smiles":"CCO", "model_type":"random_forest"}'
```

## API Documentation

FastAPI automatically generates interactive API documentation:

- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## Notebooks

The project includes Jupyter notebooks that were used to develop and evaluate the models:

- `linear_regression.ipynb`: Linear Regression model development
- `random_forest_regressor.ipynb`: Random Forest model development
- `neural_network.ipynb`: Neural Network model development

## Project Structure

```
solubility/
├── data/
│   └── delaney-processed.csv       # Dataset
├── models/                         # Saved models (created after training)
├── notebooks/
│   ├── linear_regression.ipynb
│   ├── random_forest_regressor.ipynb
│   └── neural_network.ipynb
├── src/
│   ├── app.py                      # FastAPI application
│   ├── models.py                   # ML model classes
│   ├── train_models.py             # Script to train models
│   └── utils.py                    # Utility functions
├── static/                         # Static files for web app
│   └── images/                     # Generated molecule images
├── templates/                      # HTML templates
│   └── index.html                  # Main web interface
├── requirements.txt                # Project dependencies
└── README.md                       # This file
```

## License

This project is licensed under the MIT License.
