import numpy as np
import pickle
import gensim.models.keyedvectors as word2vec

from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


with open('data/documents.pql', 'rb') as f:
    docs = pickle.load(f)
print("Number of documents:", len(docs))

model = word2vec.KeyedVectors.load_word2vec_format('ubercorpus.lowercased.lemmatized.word2vec.300d.txt', binary=False)

words = list(model.vocab)
print(words[:50], "\n\nTotal words:", len(words), "\n\nWord-Vectors shape:", model.vectors.shape)
print(model['гарний'])


def sent_embed(words, docs):
    x_sent_embed, y_sent_embed = [], []
    count_words, count_non_words = 0, 0

    # recover the embedding of each sentence with the average of the vector that composes it
    # sent - sentence, state - state of the sentence (pos/neg)
    for sent, state in docs:
        # average embedding of all words in a sentence
        sent_embed = []
        for word in sent:
            try:
                # if word is present in the dictionary - add its vector representation
                count_words += 1
                sent_embed.append(model[word])
            except KeyError:
                # if word is not in the dictionary - add a zero vector
                count_non_words += 1
                sent_embed.append([0] * 300)

        # add a sentence vector to the list
        x_sent_embed.append(np.mean(sent_embed, axis=0).tolist())

        # add a label to y_sent_embed
        if state == 'pos':
            y_sent_embed.append(1)
        elif state == 'neg':
            y_sent_embed.append(0)

    print(count_non_words, "out of", count_words, "words were not found in the vocabulary.")

    return x_sent_embed, y_sent_embed


x, y = sent_embed(words, docs)


def cosine_similarity(u, v):
    """
    Cosine similarity reflects the degree of similariy between u and v

    Arguments:
        u -- a word vector of shape (n,)
        v -- a word vector of shape (n,)

    Returns:
        cosine_similarity -- the cosine similarity between u and v defined by the formula above.
    """

    distance = 0.0

    # compute the dot product between u and v
    dot = np.dot(u, v)

    # compute the L2 norm of u
    norm_u = np.sqrt(sum(u ** 2))

    # Compute the L2 norm of v
    norm_v = np.sqrt(sum(v ** 2))

    # Compute the cosine similarity defined by formula above
    cosine_similarity = dot / (norm_u * norm_v)

    return cosine_similarity


print("Sentence #6: ", docs[6], "\n\nSentence #7: ", docs[7])
print("\nSentence Embedding #6: ", x[6], "\n\nSentence Embedding #7: ", x[7])
print("cosine_similarity = ", cosine_similarity(np.array(x[6]), np.array(x[7])))

print("Sentence #0: ", docs[0], "\n\nSentence #3: ", docs[3])
print("\nSentence Embedding #0: ", x[0], "\n\nSentence Embedding #3: ", x[3])
print("cosine_similarity = ", cosine_similarity(np.array(x[0]), np.array(x[3])))

print("Sentence #1: ", docs[1], "\n\nSentence #2: ", docs[2])
print("cosine_similarity = ", cosine_similarity(np.array(x[1]), np.array(x[2])))

# train test
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.15, random_state=1)

# train dev
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.15, random_state=1)

print('Length of x_train:', len(x_train), '| Length of y_train:', len(y_train))
print('Length of x_test:  ', len(x_test), '| Length of y_test: ', len(y_test))
print('Length of x_val:   ', len(x_val), '| Length of y_val:  ', len(y_val))
print("Shape of x_train set:", np.array(x_train).shape)

error = []

# calculating error for neighbor values between 1 and 25
for i in range(1, 25):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(x_train, y_train)
    pred_i = knn.predict(x_test)
    error.append(np.mean(pred_i != y_test))

# create KNN Classifier
knn = KNeighborsClassifier(n_neighbors=5, weights='distance')

# train the classifier using the training sets
knn.fit(x_train, y_train)

# predict the response for test dataset
y_pred = knn.predict(x_test)

print("Nearest Neighbors Result (k=5):\n" + '-' * 35)
print("Accuracy Score (k=5):", str(round(knn.score(x_test, y_test), 4) * 100) + '%')
print("Accuracy (x_train, y_train):", str(round(knn.score(x_train, y_train), 4) * 100) + '%')
print('\nClassification KNN:\n', classification_report(y_test, knn.predict(x_test)))

logit = LogisticRegression(solver='liblinear', multi_class='ovr', n_jobs=1)
logit.fit(x_train, y_train)
print("Accuracy Score:", str(round(logit.score(x_test, y_test), 4) * 100) + '%')
print('\nClassification Logistic Regression:\n', classification_report(y_test, logit.predict(x_test)))

clf = RandomForestClassifier(n_estimators=100)
clf.fit(x_train, y_train)
print("Accuracy Score:", str(round(clf.score(x_test,y_test), 4) * 100) + '%')
print('\nClassification Random Forest:\n', classification_report(y_test, clf.predict(x_test)))

with open('LogitModel.pickle', 'wb') as m:
    pickle.dump(logit, m)

with open('LogitModel.pickle', 'rb') as m:
    logit = pickle.load(m)
print("Logistic Regression Accuracy Score:", str(round(logit.score(x_test, y_test), 4) * 100) + '%')
