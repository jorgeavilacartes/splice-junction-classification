# Splice Junction Classification 
### Using deep learning
Based on Anastiasia Marzi's Thesis, from Unimib.

## `./01-data-preparation`
Selection of train, validation and test sets.
Use the notebook `01-Train-Val-Test-Sets.ipynb` to get balanced datasets. 
This will output a file inside the folder 'data', `/data/datasets.json` containing the distribution of the data for each dataset.
## `./02-model-training`

### Vector representation of a sequence `00-Embeddings.ipynb`:
1. One-Hot representation of nucleotides (class `HotEncoder`) and k-mers (class `HotEncoderKmer`)
2. Embeddings
    - Word2Vec (class `Word2Vec`)
    - DNABERT (TODO)

### Train models `01-Train.ipynb`:
1. Available architectures are included at `/src/models`, you can add as many as you want.
2. The entire training process is included in this notebook: 
    - Input representation. 
    - Model selection
    - Optimizer
    - Output interpretation

___ 
# Create environment
Create an environment called splice-junction

```
conda create --name splice-junction --file requirements.txt 
```

After clone the repo, and make sure you have properly installed dvc (included in the requirements), type 
```
dvc pull
```
to download the data into your local computer.

Tested on Ubuntu 20.04, Python 3.8.10 64-bit