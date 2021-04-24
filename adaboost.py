from math import log
from random import random


def train(train_mails, trainHam, trainSpam, sortedIGs, M):
    w = []
    h = []
    totalTrain = trainHam + trainSpam
    for i in range(0, totalTrain):
        w.append(1 / (totalTrain))
    for m in range(0, M):
        word = sortedIGs[m][0]
        # word, prediction if word exists in mail, prediction if word doesn't exist
        h.append((word, sortedIGs[m][1][1] > 0.5, sortedIGs[m][1][2] > 0.5))
        mistake = 1 / (totalTrain + 1)
        for j in range(len(train_mails)):
            if (word in train_mails[j] and train_mails[j][-1] != h[-1][1]) or (word not in train_mails[j] and train_mails[j][-1] != h[-1][2]):
                mistake += w[j]
        for j in range(len(train_mails)):
            if (word in train_mails[j] and train_mails[j][-1] == h[-1][1]) or (word not in train_mails[j] and train_mails[j][-1] == h[-1][2]):
                w[j] *= mistake/(1-mistake)
        # normalize w
        x = sum(w)
        for j in range(len(train_mails)):
            w[j] = w[j] / x
        h[m] = ((word, sortedIGs[m][1][1] > 0.5, sortedIGs[m][1][2] > 0.5), log((1-mistake)/mistake))
    return h


def predict(weighted_predictions, sortedIGs, list_of_words_in_mail, m):
    smTrue = 0
    smFalse = 0
    for i in range(m):
        if sortedIGs[i][0] in list_of_words_in_mail:
            if weighted_predictions[i][0][1]:
                smTrue += weighted_predictions[i][1]
            else:
                smFalse += weighted_predictions[i][1]
        else:
            if weighted_predictions[i][0][2]:
                smTrue += weighted_predictions[i][1]
            else:
                smFalse += weighted_predictions[i][1]
    else:
        return smTrue > smFalse
