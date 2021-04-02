<h1 align="center">Sentimento</h1>

<div align="center"> 
This repository contains 3 parts of iPython notebooks, which reveal the whole process of model development for the sentiment analysis from data processing to comparative analysis of different LSTM models. Visualization is accompanied throughout the journey. The model was created for the analysis of the Ukrainian text.
</div>

## 📥 &nbsp;Downloading Data

Before running notebooks, we first need to download all the data we will be using. 

As always, the first step is to clone the repository:

```shell
>> git clone https://github.com/JackShen1/sentimento.git
```

Learning datasets now include 1,000 positive and 1,000 negative book reviews. Originally, this data was taken from a large dataset with a review from Amazon, you can download it [here](http://www.cs.jhu.edu/~mdredze/datasets/sentiment/unprocessed.tar.gz). And then reviews of books were translated with the help of Google Translator into Ukrainian and slightly edited by me. Raw reviews can be found in the `data/` folder.

Since there is no support for the Ukrainian language in the NLTC library, we will take a different path. The most complete list of Ukrainian stop words was found [here](https://github.com/stopwords-iso/stopwords-uk) and they were used in this project.

Also at the processing stage (part 1) a stemmer was used for comparison, for good we would use PorterStemmer from nltk.stem, but for obvious reasons we can't. But this is not a problem, because writing your own PorterStemmer realization is not so difficult, so we wrote it for Python based on [this PHP code](https://github.com/teamtnt/tntsearch/blob/master/src/Stemmer/UkrainianStemmer.php).

And the last thing we need to download is a Word2Vec model. For simplicity, we will use a pretrained Word2Vec model with Ukrainian words-vectors, each of which has a dimension of 300. We chose the lematized version of this model because we already have our sample, which we processed in the part 1, which would fit perfectly here. The model can be found on [this](https://lang.org.ua/static/downloads/models/ubercorpus.lowercased.lemmatized.word2vec.300d.bz2) website. After downloading, unzip the `bz2` archive (~1Gb), for example using [this](https://www.winzip.com/win/en/bz2-file.html) application;

## 📝 &nbsp;Requirements

In order to run the iPython notebook, you'll need **Python** (`v3.6+`) and the following libraries:

- **Keras** (`v2.4+`)
- **Gensim** (`v3.8+`)
- **Pandas** (`v1.2+`)
- **NumPy** (`v1.19.5+`)
- **NLTK** (`v3.5+`)
- **python-decouple** (`v3.4+`)
- **pymorphy2-dicts-uk** (`v2.4.1+`)
- **pymorphy2** (`v0.9+`)
- **scikit-learn** (`v0.24.1`)
- **SciPy** (`v0.19.1+`)
- **Matplotlib** (`v2.1.1+`)
- **Jupyter**

The commands for installing these libraries will follow. First, let's create a virtual environment.

## 🐍 &nbsp;Creating a Virtual Environment

The easiest way to install `Keras`, `Gensim`, `NumPy`, `Jupyter`, `matplotlib` and our other libraries is to start with the Anaconda Python distribution.

1. Select your OS and follow the [installation instructions](https://docs.anaconda.com/anaconda/install/) for Anaconda Python. We recommend using Python 3.6+ (64-bit).

2. Install the Python development environment on your system:

	```shell
    >> pip install -U pip virtualenv
    ```
    
3. If you haven't done so already, download and unzip this entire repository from GitHub:
	
    ```shell
    >> git clone https://github.com/JackShen1/sentimento.git
    ```

4. Use `cd` to navigate into the top directory of the repo on your machine.

5. Open Anaconda Promt and install JupyterLab, also enter the following commands:

	```shell
    >> conda install -c conda-forge jupyterlab    # install JupyterLab
    >> conda create -n sentimento pip python=3.7  # choose the Python version
    >> source activate sentimento                 # activate the virtual environment
    ```
	
    Alternatively, you can install Jupyter with pip: `pip install jupyterlab`


6. Now we can install all the libraries we need:

	```shell
    >> pip install Keras gensim pandas numpy nltk python-decouple scikit-learn scipy matplotlib pymorphy2
    >> pip install -U pymorphy2-dicts-uk # dictionary for the Ukrainian language
    ```
   
6. Launch Jupyter by entering:
	
    ```shell
	>> jupyter notebook
	```
    

## 📋 &nbsp;Overview

In this project in 3 parts the whole process of data preparation and training of our model was described, the comparative analysis of classifiers and various models is carried out. Each stage is accompanied by data visualization. The results are good, as for such small datasets with not very accurate translation. In the future, I will expand the datasets and correct the translation. In everything else, the project works perfectly and can be easily adapted to English or Russian. Read the detailed description in notebooks.


## 📫 &nbsp;Get in touch

<p align="center">
<a href="https://www.linkedin.com/in/yevhenii-shendrikov-6795291b8/"><img src="https://img.shields.io/badge/-Jack%20Shendrikov-0077B5?style=flat&logo=Linkedin&logoColor=white"/></a>
<a href="mailto:jackshendrikov@gmail.com"><img src="https://img.shields.io/badge/-Jack%20Shendrikov-D14836?style=flat&logo=Gmail&logoColor=white"/></a>
<a href="https://www.facebook.com/jack.shendrikov"><img src="https://img.shields.io/badge/-Jack%20Shendrikov-1877F2?style=flat&logo=Facebook&logoColor=white"/></a>
<a href=""><img src="https://img.shields.io/badge/-@jackshen-0088cc?style=flat&logo=Telegram&logoColor=white"/></a>
</p>