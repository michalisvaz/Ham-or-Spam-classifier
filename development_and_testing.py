import ig_calculation
import utilities
import naiveBayes
import adaboost
import logistic_regression
import matplotlib.pyplot as plt
from os import listdir
from os.path import isfile, join

# reserve 9th folder for development and 10th for testing
print("Reading...")
trainMailsList, trainHam, trainSpam = utilities.readFiles([9, 10])

generalHyperParameters = [50, 100, 250, 500]
logregrHyperParameters = [0.1, 0.25, 0.5]

maxBayes = -1; maxlogregr = -1; maxAdaboost = -1
maxHyperParametersIndexBayes = -1; maxHyperParametersIndexAdaboost = -1; maxHyperParametersIndexlogregr = -1

allWords = {}
for mail in trainMailsList:
    if mail[-1]: #ham
        for word in mail[:-1]:
            if word in allWords.keys():
                allWords.update({word: (allWords.get(word)[0] + 1, allWords.get(word)[1] + 1, allWords.get(word)[2])})
            else:
                allWords.update({word: (1, 1, 0)})
    else:
        for word in mail[:-1]:
            if word in allWords.keys():
                allWords.update({word: (allWords.get(word)[0] + 1, allWords.get(word)[1], allWords.get(word)[2] + 1)})
            else:
                allWords.update({word: (1, 0, 1)})

allIgs = []
for word in allWords.keys():
    allIgs.append((word, ig_calculation.calculate_ig(allWords.get(word)[0], allWords.get(word)[1], allWords.get(word)[2], trainHam, trainSpam)))
sortedIGs = sorted(allIgs, key=lambda tup: tup[1][0], reverse=True)[:generalHyperParameters[-1]]                       #only keep the best max{k_i}

#read the 9th folder
mypath = join("pu_corpora_public", "pu3", "part" + "9")
development_mails = []
for file in listdir(mypath):
    with open(join(mypath, file), "r") as f:
        templist = []
        for line in f:
            for word in line.split():
                if word not in templist and word.isnumeric(): #len(word) > 1
                    templist.append(word)
    templist.append(not "spmsg" in file)      
    development_mails.append(templist)

print("Training bayes and adaboost...")
for i in range(len(generalHyperParameters)):
    print(i)
    correctBayes = 0
    correctAdaboost = 0
    basic_classifiers_with_weights = adaboost.train(trainMailsList, trainHam, trainSpam, sortedIGs, generalHyperParameters[i]) #maybe allMailsList as well
    bayes_probabilities = naiveBayes.train(sortedIGs, allWords, trainHam, trainSpam, generalHyperParameters[i])#parameters for naive_bayes
    for incoming in development_mails:
        if adaboost.predict(basic_classifiers_with_weights, sortedIGs, incoming, generalHyperParameters[i]) == incoming[-1]:
            correctAdaboost += 1
        if naiveBayes.predict(bayes_probabilities, incoming, sortedIGs, trainHam/(trainHam + trainSpam), 1 - trainHam/(trainHam + trainSpam), generalHyperParameters[i]) == incoming[-1]:
            correctBayes += 1
    if correctBayes > maxBayes:
        maxBayes = correctBayes
        maxHyperParametersIndexBayes = i
    if correctAdaboost > maxAdaboost:
        maxAdaboost = correctAdaboost
        maxHyperParametersIndexAdaboost = i

print("Training logistic regression")
for i in range(len(logregrHyperParameters)):
    print(i)
    correctlogregr = 0
    w8z = logistic_regression.train(sortedIGs, trainMailsList, logregrHyperParameters[i])
    for incoming in development_mails:
        if (logistic_regression.predict(w8z, incoming, sortedIGs) > 0.5) == incoming[-1]:
            correctlogregr += 1
    if correctlogregr > maxlogregr:
        maxlogregr = correctlogregr
        maxHyperParametersIndexlogregr = i

# now test and plot
mypath = join("pu_corpora_public", "pu3", "part" + "10")
test_mails = []
for file in listdir(mypath):
    with open(join(mypath, file), "r") as f:
        templist = []
        for line in f:
            for word in line.split():
                if word not in templist and word.isnumeric(): #len(word) > 1
                    templist.append(word)
    templist.append(not "spmsg" in file)      
    test_mails.append(templist)

totalTrain = trainHam + trainSpam
step = totalTrain // 10

greengreenBayes = []
greenBayes = []
greenredBayes = []
accuBayes = []
greengreenAdaboost = []
greenAdaboost = []
greenredAdaboost = []
accuAdaboost = []
greengreenlogregr = []
greenlogregr = []
greenredlogregr = []
acculogregr = []
i=0
for x in range(step, totalTrain, step):
    print(x)
    if x > totalTrain:
        break
    allWords = {}
    trainHam2 = 0
    trainSpam2 = 0
    for mail in trainMailsList[:x]:
        if mail[-1]: #ham
            trainHam2 += 1
            for word in mail[:-1]:
                if word in allWords.keys():
                    allWords.update({word: (allWords.get(word)[0] + 1, allWords.get(word)[1] + 1, allWords.get(word)[2])})
                else:
                    allWords.update({word: (1, 1, 0)})
        else:
            trainSpam2 += 1
            for word in mail[:-1]:
                if word in allWords.keys():
                    allWords.update({word: (allWords.get(word)[0] + 1, allWords.get(word)[1], allWords.get(word)[2] + 1)})
                else:
                    allWords.update({word: (1, 0, 1)})
    totalTrain2 = trainHam2 + trainSpam2
    allIgs = []
    for word in allWords.keys():
        allIgs.append((word, ig_calculation.calculate_ig(allWords.get(word)[0], allWords.get(word)[1], allWords.get(word)[2], trainHam, trainSpam)))
    sortedIGs = sorted(allIgs, key=lambda tup: tup[1][0], reverse=True)[:generalHyperParameters[-1]]
    
    basic_classifiers_with_weights = adaboost.train(trainMailsList[:x], trainHam2, trainSpam2, sortedIGs, generalHyperParameters[maxHyperParametersIndexAdaboost])
    bayes_probabilities = naiveBayes.train(sortedIGs, allWords, trainHam2, trainSpam2, generalHyperParameters[maxHyperParametersIndexBayes])#parameters for naive_bayes
    greengreenBayes.append([0, 0])
    greenBayes.append([0, 0])
    greenredBayes.append([0, 0])
    accuBayes.append([0, 0])
    greengreenAdaboost.append([0, 0])
    greenAdaboost.append([0, 0])
    greenredAdaboost.append([0, 0])
    accuAdaboost.append([0, 0])
    greengreenlogregr.append([0, 0])
    greenlogregr.append([0, 0])
    greenredlogregr.append([0, 0])
    acculogregr.append([0, 0])
    count = 0
    for incoming in test_mails:
        count += 1
        if naiveBayes.predict(bayes_probabilities, incoming, sortedIGs, trainHam2 / totalTrain2, 1 - (trainHam2 / totalTrain2), generalHyperParameters[maxHyperParametersIndexBayes]):
            greenredBayes[i][1] += 1
            if incoming[-1]:
                greenBayes[i][1] += 1
                accuBayes[i][1] += 1
        else:
            if not incoming[-1]:
                accuBayes[i][1] += 1
        if incoming[-1]:
            greengreenBayes[i][1] += 1
        if adaboost.predict(basic_classifiers_with_weights, sortedIGs, incoming, generalHyperParameters[maxHyperParametersIndexAdaboost]):
            greenredAdaboost[i][1] += 1
            if incoming[-1]:
                greenAdaboost[i][1] += 1
                accuAdaboost[i][1] += 1
        else:
            if not incoming[-1]:
                accuAdaboost[i][1] += 1
        if incoming[-1]:
            greengreenAdaboost[i][1] += 1
        if (logistic_regression.predict(w8z, incoming, sortedIGs) > 0.5) == incoming[-1]:
            greenredlogregr[i][1] += 1
            if incoming[-1]:
                greenlogregr[i][1] += 1
                acculogregr[i][1] += 1
        else:
            if not incoming[-1]:
                acculogregr[i][1] += 1
        if incoming[-1]:
            greengreenlogregr[i][1] += 1
    for incoming in trainMailsList[:x]:
        if naiveBayes.predict(bayes_probabilities, incoming, sortedIGs, trainHam2 / totalTrain2, trainSpam2 / totalTrain2, generalHyperParameters[maxHyperParametersIndexBayes]):
            greenredBayes[i][0] += 1
            if incoming[-1]:
                greenBayes[i][0] += 1
                accuBayes[i][0] += 1
        else:
            if not incoming[-1]:
                accuBayes[i][0] += 1
        if incoming[-1]:
            greengreenBayes[i][0] += 1
        if adaboost.predict(basic_classifiers_with_weights, sortedIGs, incoming, generalHyperParameters[maxHyperParametersIndexAdaboost]):
            greenredAdaboost[i][0] += 1
            if incoming[-1]:
                greenAdaboost[i][0] += 1
                accuAdaboost[i][0] += 1
        else:
            if not incoming[-1]:
                accuAdaboost[i][0] += 1
        if incoming[-1]:
            greengreenAdaboost[i][0] += 1
        if (logistic_regression.predict(w8z, incoming, sortedIGs) > 0.5) == incoming[-1]:
            greenredlogregr[i][0] += 1
            if incoming[-1]:
                greenlogregr[i][0] += 1
                acculogregr[i][0] += 1
        else:
            if not incoming[-1]:
                acculogregr[i][0] += 1
        if incoming[-1]:
            greengreenlogregr[i][0] += 1
    i += 1
adaboost_train_accu = []
adaboost_test_accu = []
adaboost_train_precision = []
adaboost_test_precision = []
adaboost_train_recall = []
adaboost_test_recall = []
adaboost_train_f1 = []
adaboost_test_f1 = []
for x in range(len(accuAdaboost)):
    adaboost_train_accu.append(accuAdaboost[x][0]/(step*(x+1)))
    adaboost_test_accu.append(accuAdaboost[x][1]/count)
    adaboost_train_precision.append(greenAdaboost[x][0]/greenredAdaboost[x][0])
    adaboost_test_precision.append(greenAdaboost[x][1]/greenredAdaboost[x][1])
    adaboost_train_recall.append(greenAdaboost[x][0]/greengreenAdaboost[x][0])
    adaboost_test_recall.append(greenAdaboost[x][1]/greengreenAdaboost[x][1])
    adaboost_train_f1.append(2 * (adaboost_train_precision[x] * adaboost_train_recall[x]) / (adaboost_train_precision[x] + adaboost_train_recall[x]))
    adaboost_test_f1.append(2 * (adaboost_test_precision[x] * adaboost_test_recall[x]) / (adaboost_test_precision[x] + adaboost_test_recall[x]))

Bayes_train_accu = []
Bayes_test_accu = []
Bayes_train_precision = []
Bayes_test_precision = []
Bayes_train_recall = []
Bayes_test_recall = []
Bayes_train_f1 = []
Bayes_test_f1 = []
for x in range(len(accuBayes)):
    Bayes_train_accu.append(accuBayes[x][0]/(step*(x+1)))
    Bayes_test_accu.append(accuBayes[x][1]/count)
    Bayes_train_precision.append(greenBayes[x][0]/greenredBayes[x][0])
    Bayes_test_precision.append(greenBayes[x][1]/greenredBayes[x][1])
    Bayes_train_recall.append(greenBayes[x][0]/greengreenBayes[x][0])
    Bayes_test_recall.append(greenBayes[x][1]/greengreenBayes[x][1])
    Bayes_train_f1.append(2 * (Bayes_train_precision[x] * Bayes_train_recall[x]) / (Bayes_train_precision[x] + Bayes_train_recall[x]))
    Bayes_test_f1.append(2 * (Bayes_test_precision[x] * Bayes_test_recall[x]) / (Bayes_test_precision[x] + Bayes_test_recall[x]))

logregr_train_accu = []
logregr_test_accu = []
logregr_train_precision = []
logregr_test_precision = []
logregr_train_recall = []
logregr_test_recall = []
logregr_train_f1 = []
logregr_test_f1 = []
for x in range(len(acculogregr)):
    logregr_train_accu.append(acculogregr[x][0]/(step*(x+1)))
    logregr_test_accu.append(acculogregr[x][1]/count)
    logregr_train_precision.append(greenlogregr[x][0]/greenredlogregr[x][0])
    logregr_test_precision.append(greenlogregr[x][1]/greenredlogregr[x][1])
    logregr_train_recall.append(greenlogregr[x][0]/greengreenlogregr[x][0])
    logregr_test_recall.append(greenlogregr[x][1]/greengreenlogregr[x][1])
    logregr_train_f1.append(2 * (logregr_train_precision[x] * logregr_train_recall[x]) / (logregr_train_precision[x] + logregr_train_recall[x]))
    logregr_test_f1.append(2 * (logregr_test_precision[x] * logregr_test_recall[x]) / (logregr_test_precision[x] + logregr_test_recall[x]))


percentagez = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

print( " % |      Bayes in trainset     |       Bayes in testset     ")
print("   | accur | prec | recall | f1 | accur | prec | recall | f1 ")
for x in range(len(percentagez)):
    print(percentagez[x], "|  %.3f  " % (accuBayes[x][0]/(step*(x+1))), "|  %.3f  " % (greenBayes[x][0]/greenredBayes[x][0]), 
    "|  %.3f  " % (greenBayes[x][0]/greengreenBayes[x][0]),
    "|  %.3f  " % (2 * (Bayes_train_precision[x] * Bayes_train_recall[x]) / (Bayes_train_precision[x] + Bayes_train_recall[x])),
    "|  %.3f  " % (accuBayes[x][1]/count), "|  %.3f  " % (greenBayes[x][1]/greenredBayes[x][1]), 
    "|  %.3f  " % (greenBayes[x][1]/greengreenBayes[x][1]),
    "|  %.3f  " % (2 * (Bayes_test_precision[x] * Bayes_test_recall[x]) / (Bayes_test_precision[x] + Bayes_test_recall[x])),
    )


print( " % |      Adaboost in trainset     |       Adaboost in testset     ")
print("   | accur | prec | recall | f1 | accur | prec | recall | f1 ")
for x in range(len(percentagez)):
    print(percentagez[x], "|  %.3f  " % (accuAdaboost[x][0]/(step*(x+1))), "|  %.3f  " % (greenAdaboost[x][0]/greenredAdaboost[x][0]), 
    "|  %.3f  " % (greenAdaboost[x][0]/greengreenAdaboost[x][0]),
    "|  %.3f  " % (2 * (adaboost_train_precision[x] * adaboost_train_recall[x]) / (adaboost_train_precision[x] + adaboost_train_recall[x])),
    "|  %.3f  " % (accuAdaboost[x][1]/count), "|  %.3f  " % (greenAdaboost[x][1]/greenredAdaboost[x][1]), 
    "|  %.3f  " % (greenAdaboost[x][1]/greengreenAdaboost[x][1]),
    "|  %.3f  " % (2 * (adaboost_test_precision[x] * adaboost_test_recall[x]) / (adaboost_test_precision[x] + adaboost_test_recall[x])),
    )

print( " % |      logregr in trainset     |       logregr in testset     ")
print("   | accur | prec | recall | f1 | accur | prec | recall | f1 ")
for x in range(len(percentagez)):
    print(percentagez[x], "|  %.3f  " % (acculogregr[x][0]/(step*(x+1))), "|  %.3f  " % (greenlogregr[x][0]/greenredlogregr[x][0]), 
    "|  %.3f  " % (greenlogregr[x][0]/greengreenlogregr[x][0]),
    "|  %.3f  " % (2 * (logregr_train_precision[x] * logregr_train_recall[x]) / (logregr_train_precision[x] + logregr_train_recall[x])),
    "|  %.3f  " % (acculogregr[x][1]/count), "|  %.3f  " % (greenlogregr[x][1]/greenredlogregr[x][1]), 
    "|  %.3f  " % (greenlogregr[x][1]/greengreenlogregr[x][1]),
    "|  %.3f  " % (2 * (logregr_test_precision[x] * logregr_test_recall[x]) / (logregr_test_precision[x] + logregr_test_recall[x])),
    )       


plt.plot(percentagez, adaboost_train_accu, 'r')
plt.plot(percentagez, adaboost_test_accu, 'b')
plt.title("Adaboost accuracy")
plt.xlabel("percentage of trainset")
plt.ylabel("accuracy")
plt.show()


plt.plot(percentagez, adaboost_train_precision, 'r')
plt.plot(percentagez, adaboost_test_precision, 'b')
plt.title("Adaboost precision")
plt.xlabel("percentage of trainset")
plt.ylabel("precision")
plt.show()

plt.plot(percentagez, adaboost_train_recall, 'r')
plt.plot(percentagez, adaboost_test_recall, 'b')
plt.title("Adaboost recall")
plt.xlabel("percentage of trainset")
plt.ylabel("recall")
plt.show()

plt.plot(percentagez, adaboost_train_f1, 'r')
plt.plot(percentagez, adaboost_test_f1, 'b')
plt.title("Adaboost f1 score")
plt.xlabel("percentage of trainset")
plt.ylabel("f1 score")
plt.show()


plt.plot(percentagez, Bayes_train_accu, 'r')
plt.plot(percentagez, Bayes_test_accu, 'b')
plt.title("Bayes accuracy")
plt.xlabel("percentage of trainset")
plt.ylabel("accuracy")
plt.show()

plt.plot(percentagez, Bayes_train_precision, 'r')
plt.plot(percentagez, Bayes_test_precision, 'b')
plt.title("Bayes precision")
plt.xlabel("percentage of trainset")
plt.ylabel("precision")
plt.show()

plt.plot(percentagez, Bayes_train_recall, 'r')
plt.plot(percentagez, Bayes_test_recall, 'b')
plt.title("Bayes recall")
plt.xlabel("percentage of trainset")
plt.ylabel("recall")
plt.show()

plt.plot(percentagez, Bayes_train_f1, 'r')
plt.plot(percentagez, Bayes_test_f1, 'b')
plt.title("Bayes f1 score")
plt.xlabel("percentage of trainset")
plt.ylabel("f1 score")
plt.show()

plt.plot(percentagez, logregr_train_accu, 'r')
plt.plot(percentagez, logregr_test_accu, 'b')
plt.title("logregr accuracy")
plt.xlabel("percentage of trainset")
plt.ylabel("accuracy")
plt.show()

plt.plot(percentagez, logregr_train_precision, 'r')
plt.plot(percentagez, logregr_test_precision, 'b')
plt.title("logregr precision")
plt.xlabel("percentage of trainset")
plt.ylabel("precision")
plt.show()

plt.plot(percentagez, logregr_train_recall, 'r')
plt.plot(percentagez, logregr_test_recall, 'b')
plt.title("logregr recall")
plt.xlabel("percentage of trainset")
plt.ylabel("recall")
plt.show()

plt.plot(percentagez, logregr_train_f1, 'r')
plt.plot(percentagez, logregr_test_f1, 'b')
plt.title("logregr f1 score")
plt.xlabel("percentage of trainset")
plt.ylabel("f1 score")
plt.show()