#!/usr/bin/python
#
# CIS 472/572 - Logistic Regression Template Code
#
# Author: Daniel Lowd <lowd@cs.uoregon.edu>
# Date:   2/9/2018
#
# Please use this code as the template for your solution.
#
from multiprocessing.managers import ValueProxy
import sys
import re
from math import ceil, floor, log, exp, sqrt

inputs = {}

# Load data from a file
def read_data(filename):
    f = open(filename, 'r')
    p = re.compile(',')
    header = f.readline().strip()
    varnames = p.split(header)
    exes = []
    for l in f:
        newex = []
        example = [x for x in p.split(l.strip())]
        heart_disease_flag = -1
        for var in range(len(example)):
            match varnames[var]:
                case "HeartDisease":
                    if example[var] == "Yes":
                        heart_disease_flag = 1
                    else:
                        heart_disease_flag = -1
                case "BMI": newex.append(round((float(example[var]) - 12) / 82.8, 4))
                case "Smoking" : newex.append(1) if example[var] == 'Yes' else newex.append(0)
                case "AlcoholDrinking" : newex.append(1) if example[var] == 'Yes' else newex.append(0)
                case "Stroke" : newex.append(1) if example[var] == 'Yes' else newex.append(0)
                case "PhysicalHealth" : newex.append(round(float(example[var]) / 30, 4))
                case "MentalHealth" : newex.append(round(float(example[var]) / 30, 4))
                case "DiffWalking" : newex.append(1) if example[var] == 'Yes' else newex.append(0)
                case "Sex" : newex.append(1) if example[var] == 'Male' else newex.append(0) 
                case "AgeCategory" : 
                    match example[var]:
                        case "80 or older" : newex.append(1)
                        case "75-79": newex.append(0.9167)
                        case "70-74": newex.append(0.8333)
                        case "65-69": newex.append(0.75)
                        case "60-64": newex.append(0.6667)
                        case "55-59": newex.append(0.5833)
                        case "50-54": newex.append(0.5)
                        case "45-49": newex.append(0.4167)
                        case "40-44": newex.append(0.3333)
                        case "35-39": newex.append(0.25)
                        case "30-34": newex.append(0.1667)
                        case "25-29": newex.append(0.0833)
                        case "18-24": newex.append(0)
                        case _ : print(example[var])
                case "Race" : 
                    match example[var]:
                        case "Asian": newex.append(1)
                        case "Hispanic": newex.append(0.75)
                        case "Black": newex.append(0.5)
                        case "White": newex.append(0.25)
                        case _ : newex.append(0)
                case "Diabetic" :
                    match example[var]:
                        case "Yes": newex.append(1)
                        case "Borderline diabetes": newex.append(0.6667)
                        case "Yes (during pregnancy)": newex.append(0.3333)
                        case "No": newex.append(0)
                        case _ : print(example[var])
                case "PhysicalActivity" : newex.append(1) if example[var] == 'Yes' else newex.append(0)
                case "GenHealth" : 
                    match example[var]:
                        case "Excellent": newex.append(1)
                        case "Very good": newex.append(0.75)
                        case "Good": newex.append(0.5)
                        case "Fair": newex.append(0.25)
                        case "Poor": newex.append(0)
                        case _ : print(example[var])
                case "SleepTime" : newex.append(round(float(example[var]) / 24, 4))
                case "Asthma" : newex.append(1) if example[var] == 'Yes' else newex.append(0)
                case "KidneyDisease" : newex.append(1) if example[var] == 'Yes' else newex.append(0)
                case "SkinCancer": newex.append(1) if example[var] == 'Yes' else newex.append(0)
                case _ : print("Error")
        newex.append(heart_disease_flag)
        for el in range(len(newex)):
            newex[el] = round(newex[el], 4)
        exes.append(newex)
    return exes, varnames

def writeLine(line, file):
    for i in range(len(line)):
        if not i == len(line) - 1:
            file.write('%s,' % line[i])
        else:
            file.write('%s\n' % line[i])


# Load train and test data.  Learn model.  Report accuracy.
def main(argv):
    if (len(argv) != 5):
        print('Usage: lr.py <train> <dataFile>')
        sys.exit(2)
    conv_examples, varnames = read_data(argv[0])
    dataFile = argv[1]
    dataTrain = argv[2]
    dataValid = argv[3]
    dataTest = argv[4]
    datalen = len(conv_examples)
    f = open(dataFile, "w+")
    train = open(dataTrain, "w+")
    valid = open(dataValid, "w+")
    test = open(dataTest, "w+")
    # Write model file
    # This would be far prettier as a dictionary - if extra time permits, refactor for cleanliness of code
    # Current implementation is clunky and hard to follow
    filedict = {
        'f': [f, conv_examples[1:]],
        'train': [train, conv_examples[1:floor(datalen * 0.7)]],
        'valid': [valid, conv_examples[ceil(datalen * 0.7):floor(datalen * 0.9)]],
        'test': [test, conv_examples[ceil(datalen * 0.9):]]
    }
    for file in filedict:
        writeLine(varnames, filedict[file][0])
        for ex in filedict[file][1]:
            writeLine(ex, filedict[file][0])

if __name__ == "__main__":
    main(sys.argv[1:])
