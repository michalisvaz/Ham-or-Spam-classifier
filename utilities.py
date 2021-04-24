from os import listdir
from os.path import isfile, join
from ig_calculation import calculate_ig
import string

# returns list_of_all_mails, total ham mails, total spam mails
# each item in the list_of_all_mails is a list(which represents a mail)
# containing each of the mail's words(only once) and at the last place a value
# True/False for ham/spam respectively

def readFiles(exceptWhich):
    allmails = []
    totalspam = 0
    totalham = 0
    for i in range(1, 11):
        if i not in exceptWhich:
            mypath = join("pu_corpora_public", "pu3", "part" + str(i))
            for file in listdir(mypath):
                with open(join(mypath, file), "r") as f:
                    templist = []
                    for line in f:
                        for word in line.split():
                            if word not in templist and word.isnumeric(): #len(word) > 1
                                templist.append(word)
                if "spmsg" in file: #spam --> False
                    totalspam += 1
                    templist.append(False)
                else: #ham --> True
                    totalham += 1
                    templist.append(True)            
                allmails.append(templist)
    return allmails, totalham, totalspam