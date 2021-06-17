import os
import glob
import importlib
from pathlib import Path
import numpy as np

OMMIT = {".ipynb_checkpoints","__pycache__","__init__"}
BASE_DIR = Path(__file__).resolve().parent

class ModelLoader:
    "Clase para cargar modelos definidos dentro de '/models'"
    AVAILABLE_MODELS = [model[:-3] for model in os.listdir(BASE_DIR.joinpath('models')) if all([ommit not in model for ommit in OMMIT])]
    BASE_DIR = BASE_DIR

    def __call__(self, 
            model_name, 
            n_output,
            output_layer,
            weights_path=None,
            ):
        # Comprobar que el modelo llamado existe
        assert model_name in self.AVAILABLE_MODELS, f"'model_name' debe ser uno de ellos {self.AVAILABLE_MODELS}.El suyo es {model_name!r}"

        # Cargar la clase para
        ModelDNA = getattr(
            importlib.import_module(
                f"src.models.{model_name}"
            ),
            "ModelDNA"
        )

        # Instanciar clase para extraer arquitectura
        model_dna = ModelDNA(n_output, output_layer,) 
        
        # cargar arquitecture
        model = model_dna.get_model()

        # Cargar pesos si es se provee
        if weights_path is not None:
            print(f"\n Cargando pesos desde weights_path** : {weights_path}")
            model.load_weights(weights_path)
        print(f"Modelo {model_name!r} cargado correctamente")
        return model