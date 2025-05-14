import py3Dmol
from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.metrics import root_mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

def show_molecule_from_smiles(smiles: str):
    """
    Visualize the 3D structure of the molecule
    starting from the smiles convention.

    Args:
        - smiles [str]: representation of the molecule
    """
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)

    AllChem.EmbedMolecule(mol, AllChem.ETKDG())
    AllChem.UFFOptimizeMolecule(mol)

    block = Chem.MolToMolBlock(mol)
    
    viewer = py3Dmol.view(width=400, height=400)
    viewer.addModel(block, "mol")
    viewer.setStyle({'stick': {}, "sphere": {"radius": 0.4}})
    viewer.setBackgroundColor('white')
    viewer.zoomTo()
    viewer.show()

def show_accuracy(y_test, y_pred):
    """
    Print the scores R-squared and Root Mean Square Error
    and plot the predicted values against the actual values.

    Args:
        - y_test: actual values
        - y_pred: predicted values
    """
    r2 = r2_score(y_test, y_pred)
    rmse = root_mean_squared_error(y_test, y_pred)

    print(f"R^2: {r2:.4f}")
    print(f"RMSE: {rmse:.4f}")

    plt.figure()
    sns.scatterplot(x=y_test, y=y_pred)
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--')
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.grid()
    plt.show()