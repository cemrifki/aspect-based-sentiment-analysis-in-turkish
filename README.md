# Aspect-Based Sentiment Analysis for Turkish 

This approach attempts to perform aspect-based sentiment analysis (ABSA). This extracts aspects and then finds their sentiments. The corresponding data is received from: https://www.kaggle.com/code/nkitgupta/aspect-based-sentiment-analysis/. I have also updated the relevant code a lot, relying on other models (e.g., Sentence Transformers) for embeddings, different classifiers, further preprocessing techniques / rules and metrics as well. In the end, I could boost the performance even further by leveraging the SVM classifier, whose computation complexity is lower as compared to those of BERT models as is known. The code is runnable for both English and Turkish. This repo can be adapted to other languages as well by performing minor tweaks, such as changing the corresponding SBERT model and using the related spaCy model and the corresponding dataset for the specific language.

## Requirements

- Python 3.9 (or a newer version)
- pandas
- torch
- transformers
- spacy
- thinc
- scipy
- scikit-learn 
- gensim
- sentence_transformers
- nltk
- numpy
- huggingface-hub
- lxml (optional)
- xgboost (optional)

 The code can work through `Python 3.9` or a newer version thereof. I tested it relying on `Python 3.9`. In this project, `python3` and `pip3`. We leveraged two datasets, which are sentiment and spam corpora and which can be found in the input folder.
 
 ## Execution

Execute the file `main.py` to train the model and evaluate it on the test partition of the very same dataset. You can set the name of the SBERT model, the overall number of corpus aspects, the path to the input .csv file, and the name of the text column in the `constants.py` file, which can be considered to be the config file.

#### Setup with virtual environment (Python 3):

-  `python3 -m venv my_venv`
-  `source my_venv/bin/activate`

Install the requirements:

-  `pip3 install -r requirements.txt`

If everything works well, you can run the example usage given below.

### Example Usage:

- The following guide shows an example usage of the model performing training and evaluation for the aspect-based sentiment analysis task.
- Instructions
      
      1. Change directory to the location of the source code
      2. Run the instructions in "Setup with virtual environment (Python 3)"
      3. Run the exemplary main.py file.

Example:
```
python3 main.py
```
## Citation
If you find this code useful, please cite the following in your work:
```
@phdthesis{cra:20,
  author       = {Cem Rifki Aydin}, 
  title        = {Developing a Comprehensive Framework for Sentiment Analysis in Turkish},
  school       = {Bogazici University},
  year         = 2020
}
```
## Credits
- The code has been written by Cem Rifki Aydin
- Also, the dataset is received from: https://www.kaggle.com/code/nkitgupta/aspect-based-sentiment-analysis/output