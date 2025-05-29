# Aspect-Based Sentiment Analysis for Turkish 

This approach attempts to perform aspect-based sentiment analysis (ABSA) for English first. This extracts aspects and then finds their sentiments. The corresponding data is received from: https://www.kaggle.com/code/nkitgupta/aspect-based-sentiment-analysis/notebook. I have also updated the relevant code a lot, relying on other models (e.g., Sentence Transformers) for embeddings, different classifiers, further preprocessing techniques / rules and metrics as well. In the end, I could boost the performance even further by leveraging the SVM classifier, whose computation complexity is lower as compared to BERT models as known. The corresponding code for Turkish will be available soon as well. However, anyway, this repo can be made language-agnostic by performing minor tweaks, such as changing the SBERT model and using the related spaCy model for a specific language.

## Requirements

- Python 3.8 or a newer version
- pandas
- torch
- transformers
- spacy
- scikit-learn 
- gensim
- sentence_transformers
- nltk
- numpy

 The code can work through `Python 3.8` or a newer version thereof. I tested it relying on `Python 3.12`. In this project, `python3` and `pip3`. We leveraged two datasets, which are sentiment and spam corpora and which can be found in the input folder.
 
 ## Execution

Execute the file `main.py` to train the model and evaluate it on the test partition of the very same dataset. You can set the overall number of corpus aspects and the path to the input .csv file in the `constants.py` file, which can be considered as a config file.

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
- Also, the dataset is received from: https://www.kaggle.com/code/nkitgupta/aspect-based-sentiment-analysis/notebook