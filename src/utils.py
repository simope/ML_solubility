import py3Dmol
from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.metrics import root_mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import uuid

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

def get_3d_molecule_html(smiles: str):
    """
    Generate HTML for 3D visualization of a molecule from SMILES.
    
    Args:
        - smiles [str]: representation of the molecule
        
    Returns:
        - str: HTML code for 3D visualization that can be embedded in a web page
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return "<div>Invalid SMILES string</div>"
    
    mol = Chem.AddHs(mol)
    
    try:
        AllChem.EmbedMolecule(mol, AllChem.ETKDG())
        AllChem.UFFOptimizeMolecule(mol)
    except:
        return "<div>Failed to generate 3D coordinates</div>"
    
    block = Chem.MolToMolBlock(mol)
    
    # Generate a unique ID for the viewer div
    viewer_id = f"molecule_viewer_{uuid.uuid4().hex[:12]}"
    
    html = f"""
    <div id="{viewer_id}" style="position: relative; width: 400px; height: 400px; margin: 0 auto;"></div>
    <script>
        var loadScript = function(url, callback) {{
            var script = document.createElement("script");
            script.type = "text/javascript";
            script.src = url;
            script.onload = callback;
            document.head.appendChild(script);
        }};
        
        if (typeof $3Dmol === 'undefined') {{
            loadScript('https://cdnjs.cloudflare.com/ajax/libs/3Dmol/2.0.1/3Dmol-min.js', function() {{
                createViewer();
            }});
        }} else {{
            createViewer();
        }}
        
        function createViewer() {{
            let viewer = $3Dmol.createViewer(document.getElementById('{viewer_id}'), {{backgroundColor: 'white'}});
            viewer.addModel(`{block}`, 'mol');
            viewer.setStyle({{'stick': {{}}, 'sphere': {{'radius': 0.4}}}});
            viewer.zoomTo();
            viewer.render();
        }}
    </script>
    """
    
    return html

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