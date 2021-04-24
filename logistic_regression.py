import numpy as np
import random

def train(sortedigs, train_examples, lnorm, howmany = 200, trainsize = 100):
    w = []
    for i in range(howmany):
        w.append(random.random())
    s = 0
    e = 0.1
    swanted = 420
    count = 0
    while(s < swanted):
        count  += 1
        s = 0
        probs = []
        for j in range(trainsize):
            L = 0
            for i in range(howmany):
                prob = predict(w, train_examples[j], sortedigs)
                probs.append(prob)
                if(train_examples[j][-1]):
                    L += prob + 1/trainsize
                else:
                    L += (1 - prob) + 1/trainsize
            L = np.log(L)
            summ = 0
            for i in w:
                summ += i * i
            L = L - lnorm * summ
            s += L
            for i in range(howmany):
                if(train_examples[j][-1]):
                    y = 1
                else:
                    y = 0
                if sortedigs[i][0] in train_examples[j]:
                    x = 1 - 1/trainsize
                else:
                    x = 1/trainsize
                w[i] = w[i] + e * (1 / (x * (y - probs[i] + 1 / trainsize))) * (y - probs[i]) * x - 2 * lnorm * w[i]
        e *= 0.8
        if(count > 10):
            break
    return w



def predict(w, x, sortedigs):
    t = 0
    for i in range(len(w)):
        if(sortedigs[i][0] in x): #word
            t += w[i]
    prob = 1 / (1 + np.exp(-t))
    return prob