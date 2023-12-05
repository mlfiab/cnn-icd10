# Automatic ICD-10 code classification system in French with CNN
- [Supervised Learning for the ICD-10 Coding of French Clinical Narratives](https://hal.science/hal-03020990/file/MIE2020.pdf)


## Reference
Please cite the following paper:
```
    @article{bouzille2020supervised,
    title={Supervised learning for the ICD-10 coding of French clinical narratives},
    author={BOUZILLE, Guillaume and GRABAR, Natalia},
    journal={Digital Personalized Health and Medicine: Proceedings of MIE 2020},
    volume={270},
    pages={427},
    year={2020},
    publisher={IOS Press}
    }
```


## Requirements
* Python >= 3.6 (recommended via anaconda)
* Install the required Python packages with `pip install -r requirements.txt`
* If the specific versions could not be found in your distribution, you could simple remove the version constraint. Our code should work with most versions.

## Dataset
Obviously for privacy reasons, we are not allowed to share the dataset used in this work. For execution you have to put your data in `data` folder

We assume that the dataset contain the columns : 'text' & 'CIM10'
1. Column 'text': Input data containing text medical data 
2. Column 'CIM10': ICD codes list
e.g: ['E86', 'J100', 'E8708', 'J90']

## Architectures
1. FastText embedding with skip-gram algorithm
2. CNN model for text classification 
## How to run

## First, Embedding Training 
1. Provide a text file for embedding in the folder (`embed/data.txt`)
2. Run the following command to train the embedding 

```
    python fastest_embed.py --train_file embed/data.txt --embed_output fasttext_300D
```

### Model Training
1. Put the dataset in the folder: `data`
2. Run the following command to train the model.

```
    python main.py --train_file data/DATA.csv --embed_input fasttext_300D
```

### Notes
- You can change the default parameters of the CNN architecture used in `models.py` file or the hyper-parameters in `main.py`
