import pickle
import pymorphy2
import numpy as np
import pandas as pd
import nltk.classify.util
import xml.etree.ElementTree as ET


from re import match
from nltk import FreqDist
from random import shuffle
from stemmer_ua import UAStemmer
from nltk.classify import NaiveBayesClassifier
from sklearn.model_selection import train_test_split
from nltk.tokenize import word_tokenize, sent_tokenize, RegexpTokenizer

testRead = open('data/positive.review', 'r', encoding='utf8', errors='ignore').readlines()
print('===========TEXT EXAMPLE===========')
for i in range(1, 44):
    print(testRead[i])
print('==================================')

pos_reviews = open('data/positive.review', 'r', encoding='utf8', errors='ignore').read()
neg_reviews = open('data/negative.review', 'r', encoding='utf8', errors='ignore').read()

pos_rev_tree = ET.fromstring(pos_reviews)
neg_rev_tree = ET.fromstring(neg_reviews)

pos_tags = pos_rev_tree.findall('review')
neg_tags = neg_rev_tree.findall('review')

print('\nNumber of Positive Reviews:', len(pos_tags),
      '\nNumber of Positive Reviews:', len(neg_tags))

REVIEW_TAGS = ['unique_id', 'asin', 'product_name', 'helpful', 'rating', 'title',
               'date', 'reviewer', 'reviewer_location', 'review_text']


def parseXML(xml_reviews):
    reviews = {}
    count = 0

    for item in xml_reviews:
        count += 1
        rev_name = 'review' + str(count)
        reviews[rev_name] = [
            REVIEW_TAGS[0] + ' | ' + item.find(REVIEW_TAGS[0]).text.strip(),
            REVIEW_TAGS[1] + ' | ' + item.find(REVIEW_TAGS[1]).text.strip(),
            REVIEW_TAGS[2] + ' | ' + item.find(REVIEW_TAGS[2]).text.strip(),
            REVIEW_TAGS[3] + ' | ' + item.find(REVIEW_TAGS[3]).text.strip(),
            REVIEW_TAGS[4] + ' | ' + item.find(REVIEW_TAGS[4]).text.strip(),
            REVIEW_TAGS[5] + ' | ' + item.find(REVIEW_TAGS[5]).text.strip(),
            REVIEW_TAGS[6] + ' | ' + item.find(REVIEW_TAGS[6]).text.strip(),
            REVIEW_TAGS[7] + ' | ' + item.find(REVIEW_TAGS[7]).text.strip(),
            REVIEW_TAGS[8] + ' | ' + item.find(REVIEW_TAGS[8]).text.strip(),
            REVIEW_TAGS[9] + ' | ' + item.find(REVIEW_TAGS[9]).text.strip()
        ]

    return reviews


pos_reviews_dict = parseXML(pos_tags)
neg_reviews_dict = parseXML(neg_tags)


def dict_to_dataframe(reviews_dict):
    # prepare our dataframe for the data
    df = pd.DataFrame(columns=REVIEW_TAGS)
    count = 0
    for val in reviews_dict.values():
        df.loc[count] = [
            val[0].split("|")[1], val[1].split("|")[1],
            val[2].split("|")[1], val[3].split("|")[1],
            val[4].split("|")[1], val[5].split("|")[1],
            val[6].split("|")[1], val[7].split("|")[1],
            val[8].split("|")[1], val[9].split("|")[1]
        ]

        count = count + 1

    return df


pos_books = dict_to_dataframe(pos_reviews_dict)
neg_books = dict_to_dataframe(neg_reviews_dict)

print('\nNumber of Positive Reviews (Dataframe):', len(pos_books),
      '\nNumber of Negative Reviews (Dataframe):', len(neg_books))

pos_books.drop('unique_id', axis=1, inplace=True)
neg_books.drop('unique_id', axis=1, inplace=True)

pos_books['Class'] = "pos"  # positive reviews
neg_books['Class'] = "neg"  # negative reviews

reviews = pd.concat([pos_books, neg_books])

print('\n===========REVIEW TEXT EXAMPLE===========\n' +
      reviews.iloc[1, 8] +
      '\n=========================================\n')


def word_sent_tokenize(sent):
    return word_tokenize(sent), sent_tokenize(sent)


def regex_tokenizer(sent):
    return RegexpTokenizer(r'\w+').tokenize(sent)


words, sents = word_sent_tokenize(reviews.iloc[1, 8])
print("\tReview ID1 - Words:\n", words, "\n\n\tReview ID1 - Sentences:\n", sents)


STOP_WORDS = set(line.strip() for line in open('data/stopwords_ua.txt', mode="r", encoding="utf-8"))


def stopwords_elimination(stop_words, sent):
    return [w for w in sent if w not in stop_words]


filtered_words = stopwords_elimination(STOP_WORDS, words)
print("\n==================== WORDS FROM REVIEW 1 ====================\n", words,
      "\n\n================== FILTERED WORDS REVIEW 1==================\n", filtered_words,
      "\n\n======================== STOP WORDS ========================\n", STOP_WORDS)


def stem_words(words):
    return [UAStemmer(word).stem_word() for word in words]


stemmed_words = stem_words(filtered_words)
print(stemmed_words)

morph = pymorphy2.MorphAnalyzer(lang='uk')
print(morph.parse('відчувається'))


def lemmatize_words(morph, words):
    return [morph.parse(word)[0].normal_form for word in words]


lemmatized_words = lemmatize_words(morph, filtered_words)
print(lemmatized_words)

# Use regex_tokenizer to get rid of punctuation
words_list = [regex_tokenizer(item) for item in list(reviews['review_text'])]
print("Review without punctuation: ", words_list[1])

# Eliminate stopwords
words_list = [stopwords_elimination(STOP_WORDS, word) for word in words_list]
print("\n\nReview after stopwords elimination: ", words_list[1])

# Lemmatizing
words_list = [lemmatize_words(morph, word) for word in words_list]
print("\n\nReview after lemmatizing: ", words_list[1])

all_words = np.concatenate(words_list)
all_words_freq = FreqDist(all_words)  # count the number of times that each outcome of an experiment occurs

print("Most popular words: ", all_words_freq.most_common(100))
print("\nTotal number of words: ", len(all_words_freq.keys()))


def irr_words_elim(words):
    for word in words:
        if bool(match(r"[a-zA-Z0-9]", word)):
            words.remove(word)
        elif len(word) <= 3:
            words.remove(word)

    return words


for review in words_list:
    irr_words_elim(review)

all_words = np.concatenate(words_list)
all_words_freq = FreqDist(all_words)

print("\nTotal number of words: ", len(all_words_freq))


def create_doc(reviews, words_list):
    category = list(reviews['Class'])
    docs = [(words_list[word], category[word]) for word in range(len(words_list))]
    shuffle(docs)
    return docs


docs = create_doc(reviews, words_list)
print(docs[1], "\n\nNumber of documents:", len(docs))


def find_matches(doc, all_words, num_freq=50):
    word_matches = [word[0] for word in all_words.most_common(num_freq)]
    words = regex_tokenizer(doc)

    matches = {}
    for word in word_matches:
        matches[word] = (word in words)

    return matches


print(find_matches("У мене є книга, яку я полюбляю читати коли вечоріє...", all_words_freq))


def create_matches_set(docs, all_words, num_freq=100):
    matches = []
    for doc, state in docs:
        match = find_matches(' '.join(doc), all_words, num_freq)
        matches.append((match, state))
    return matches


matches_set = create_matches_set(docs[:10], all_words_freq)
print(matches_set[1])

train_set, test_set = train_test_split(matches_set, random_state=42)
print(train_set[5])

with open("output/matches_set.pql", "wb") as ms:   # Pickling
    pickle.dump(matches_set, ms)

reviews.to_csv("output/reviews.csv", sep='\t', encoding='utf-8')

with open("output/matches_set.pql", "rb") as ms:   # Unpickling
    r = pickle.load(ms)
print(r[2])

matches_set = create_matches_set(docs, all_words_freq, 2500)

train_set, test_set = train_test_split(matches_set, test_size=0.33, random_state=42)
print("Length of Matches Set:", len(matches_set), "\nLength of Train Set:", len(train_set))


def naive_bayes_model(train_set, test_set):
    classifier = NaiveBayesClassifier.train(train_set)
    informative = classifier.show_most_informative_features(10)
    accuracy = nltk.classify.accuracy(classifier, test_set) * 100
    return classifier, informative, accuracy


classifier, informative, accuracy = naive_bayes_model(train_set, test_set)
print(informative)
print("\nClassifier Accuracy:", str(round(accuracy, 2)) + "%")  # predict


def words_preparation(sent):
    words = regex_tokenizer(sent)                     # remove punctuation
    words = stopwords_elimination(STOP_WORDS, words)  # eliminate stopwords
    words = lemmatize_words(morph, words)             # lemmatizing
    words = irr_words_elim(words)                     # irrelevent words elimination
    print("Sentence after processing:", words)
    return " ".join(words)


test_sent = "Це найкращий фільм, який я бачив!"
print("Test sentence:", test_sent)

test_sent = find_matches(words_preparation(test_sent), all_words_freq, 2500)
print(classifier.classify(test_sent))

test_sent = "Ця книга є достатньо читабельною.."
print("Test sentence:", test_sent)

test_sent = find_matches(words_preparation(test_sent), all_words_freq, 2500)
print(classifier.classify(test_sent))

test_sent = "Мене дратує, що знову ввели локдаун, це неправильний підхід."
print("Test sentence:", test_sent)

test_sent = find_matches(words_preparation(test_sent), all_words_freq, 2500)
print(classifier.classify(test_sent))
