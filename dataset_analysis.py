import os
import numpy as np
import pandas as pd
import xml.etree.ElementTree as ET
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
