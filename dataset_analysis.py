import os
import sklearn
import numpy as np
import pandas as pd
import seaborn as sns
import xml.etree.ElementTree as ET


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

review_tags = ['unique_id', 'asin', 'product_name', 'helpful', 'rating', 'title',
               'date', 'reviewer', 'reviewer_location', 'review_text']


def parseXML(xml_reviews):
    reviews = {}
    count = 0

    for item in xml_reviews:
        count += 1
        rev_name = 'review' + str(count)
        reviews[rev_name] = [
            review_tags[0] + ' | ' + item.find(review_tags[0]).text,
            review_tags[1] + ' | ' + item.find(review_tags[1]).text,
            review_tags[2] + ' | ' + item.find(review_tags[2]).text,
            review_tags[3] + ' | ' + item.find(review_tags[3]).text,
            review_tags[4] + ' | ' + item.find(review_tags[4]).text,
            review_tags[5] + ' | ' + item.find(review_tags[5]).text,
            review_tags[6] + ' | ' + item.find(review_tags[6]).text,
            review_tags[7] + ' | ' + item.find(review_tags[7]).text,
            review_tags[8] + ' | ' + item.find(review_tags[8]).text,
            review_tags[9] + ' | ' + item.find(review_tags[9]).text
        ]

    return reviews


posReviewsDict = parseXML(pos_tags)
negReviewsDict = parseXML(neg_tags)


def dictToDataFrame(reviews_dict):
    # prepare our dataframe for the data
    df = pd.DataFrame(columns=review_tags)
    count = 0
    for val in reviews_dict.values():
        df.loc[count] = [
            val[0].split("|")[1].split("\n")[1], val[1].split("|")[1].split("\n")[1],
            val[2].split("|")[1].split("\n")[1], val[3].split("|")[1].split("\n")[1],
            val[4].split("|")[1].split("\n")[1], val[5].split("|")[1].split("\n")[1],
            val[6].split("|")[1].split("\n")[1], val[7].split("|")[1].split("\n")[1],
            val[8].split("|")[1].split("\n")[1], val[9].split("|")[1].split("\n")[1]
        ]

        count = count + 1

    return df


posBooks = dictToDataFrame(posReviewsDict)
negBooks = dictToDataFrame(negReviewsDict)

posBooks.head(n=3)
