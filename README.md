# Molecule Solubility Prediction

This project explores the prediction of molecular solubility using various machine learning models, including linear regression, random forest regression, and neural networks. The dataset used is based on the Delaney solubility dataset, with engineered features for improved model performance.

## Project Structure

- `data/delaney-processed.csv`: The main dataset containing molecular features and solubility values.
- `src/utils.py`: Utility functions for molecule visualization and model evaluation.
- `linear_regression.ipynb`: Jupyter notebook implementing linear regression for solubility prediction.
- `random_forest_regressor.ipynb`: Jupyter notebook using a random forest regressor.
- `neural_network.ipynb`: Jupyter notebook implementing a neural network with PyTorch.
- `requirements.txt`: List of required Python packages.

## Dataset

The dataset (`data/delaney-processed.csv`) contains the following columns:
- Compound ID
- ESOL predicted log solubility in mols per litre
- Minimum Degree
- Molecular Weight
- Number of H-Bond Donors
- Number of Rings
- Number of Rotatable Bonds
- Polar Surface Area
- Measured log solubility in mols per litre (target)
- SMILES (molecular structure representation)

## Notebooks

- **Linear Regression**: Baseline model using scikit-learn to predict solubility.
- **Random Forest Regressor**: Ensemble model for improved accuracy.
- **Neural Network**: Deep learning approach using PyTorch, with feature engineering and data scaling.

## Utilities

- `show_molecule_from_smiles(smiles)`: Visualizes a molecule in 3D from its SMILES string.
- `show_accuracy(y_test, y_pred)`: Prints R² and RMSE, and plots predicted vs. actual values.

## Installation

1. Clone the repository.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. (Optional) Set up a virtual environment for isolation.

## Usage

Open any of the Jupyter notebooks and run the cells to train and evaluate the models. You can visualize molecules and model performance using the provided utility functions.

## Requirements

- Python 3.8+
- See `requirements.txt` for all dependencies (notably: scikit-learn, pandas, numpy, matplotlib, seaborn, rdkit, py3Dmol, torch, torchvision).

## Acknowledgements

- Dataset: [Delaney, J. S. (2004). ESOL: Estimating aqueous solubility directly from molecular structure. Journal of Chemical Information and Computer Sciences, 44(3), 1000-1005.](https://pubs.acs.org/doi/10.1021/ci034243x)
- RDKit and Py3Dmol for cheminformatics and visualization.
