import numpy as np
from collections import OrderedDict as OrderedDict
from os import listdir
from os.path import join
import utilities as util
from utilities import readFiles
from ig_calculation import calculate_ig
import numpy as np

def train(sortedIgs, allwords, totalHam, totalSpam, howmanyIgs):
    probabilities = []
    for word in sortedIgs:
        P_x1_c1 = (allwords.get(word[0])[1] + 1) / (totalHam + 2)
        P_x0_c1 = (totalHam - allwords.get(word[0])[1] + 1) / (totalHam + 2)
        P_x1_c0 = (allwords.get(word[0])[2] + 1) / (totalSpam + 2)
        P_x0_c0 = (totalSpam - allwords.get(word[0])[2] + 1) / (totalSpam + 2)
        probabilities.append((P_x1_c1, P_x0_c1, P_x1_c0, P_x0_c0))
    return probabilities

def predict(probabilities, listOfWords, sortedIgs, P_c1, P_c0, howmany):
    p_ham = np.log2(P_c1)
    p_spam = np.log2(P_c0)
    for i in range(howmany):
        if sortedIgs[i][0] in listOfWords:
            p_ham += np.log2(probabilities[i][0])
            p_spam += np.log2(probabilities[i][2])
        else:
            p_ham += np.log2(probabilities[i][1])
            p_spam += np.log2(probabilities[i][3])
    return p_ham > p_spam