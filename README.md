# Aspect-Based Sentiment Analysis for Turkish 

This approach attempts to perform aspect-based sentiment analysis (ABSA). This extracts aspects and then finds their sentiments. I rely on both unsupervised and supervised models and techniques. The corresponding data is received from: https://www.kaggle.com/code/nkitgupta/aspect-based-sentiment-analysis/ for the unsupervised approach, where no aspect terms are known, but sentiments are given. I have also updated the relevant code a lot, relying on other models (e.g., Sentence Transformers) for embeddings, different classifiers, further preprocessing techniques / rules and metrics as well. In the end, I could boost the performance even further by leveraging the SVM classifier, whose computation complexity is lower as compared to those of BERT models as is known. On the other hand, for the supervised approach, I used the SemEval datasets, which provide aspect categories along with their corresponding sentiments. I extended the BERT model into a deep neural network to perform this task. The code is runnable for both English and Turkish. This repo can be adapted to other languages as well by performing minor tweaks, such as changing the corresponding SBERT model and using the related spaCy model and the corresponding dataset for the specific language.

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
- lxml

 The code can work through `Python 3.9` or a newer version thereof. I tested it relying on `Python 3.9`. In this project, `python3` and `pip3`. We leveraged two datasets, which are sentiment and spam corpora and which can be found in the input folder.
 
 ## Execution

Execute the file `main.py` to train the model and evaluate it on the test partition of the very same dataset. You can set the language parameter, the overall number of corpus aspects, the path to the input data files, and other parameters in the command line, based on the argument parser. I provided several exemplary commands towards the end of this README.md document.

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

Example for training the unsupervised approach (i.e., LDA) for English:

```
python3 main.py \
    --mode unsupervised \
    --lang en \
    --dataset data/english/clean_data.csv \
    --text clean_review \
    --output_dir outputs/unsupervised/en 
```

In order to predict a text sample, run the below:

```
python3 main.py --predict \ 
    --mode unsupervised \ 
    --lang en \ 
    --model_dir outputs/unsupervised/en \ 
    --text "Its very nasty product thank you flipkart i have used its very nasty settings and camera is also bad battery is 
    moreover the phone is evil for the money. "
```

For the supervised approach, where aspects and their polarities are both given in the dataset, you can run the below to
train the model:

```
python3 main.py \
    --mode supervised \
    --lang en \
    --dataset data/english/restaurant_reviews.xml \
    --output_dir outputs/supervised/en \
    --epochs 3 --batch_size 4
```

Apart from these, to run the supervised approach for Turkish, please run the exemplary command or another similar one:

```
python3 main.py \
    --mode supervised \
    --lang tr \
    --dataset data/turkish/restaurant_reviews.xml \
    --output_dir outputs/supervised/tr \
    --epochs 5 --batch_size 4
```

As another example, after training the unsupervised approach, you can predict aspects along with their sentiments for the Turkish
language as follows:

```
python3 main.py --predict \
    --mode unsupervised \
    --lang tr \
    --model_dir outputs/unsupervised/tr \
    --text "Yemek oldukça kötü idi. :'(("
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
- Also, the dataset for the unsupervised approach is received from: https://www.kaggle.com/code/nkitgupta/aspect-based-sentiment-analysis/output