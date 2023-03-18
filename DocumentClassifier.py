from sklearn.datasets import fetch_20newsgroups  # load data from sklearn dataset
import seaborn as sns

data=fetch_20newsgroups()
categories=data.target_names
# Training the data on these categories
train=fetch_20newsgroups(subset='train', categories=categories)
# Testing the data for these categories
test=fetch_20newsgroups(subset='test', categories=categories)

def preProcessing(text):
    import re
    import string
    # Convert text to lowercase
    outText = text.lower()
    
    # Remove numbers
    outText = re.sub(r'\d+', '', outText)
    
    # Remove punctuation
    outText =  outText.translate(str.maketrans("","", string.punctuation))
    
    #Remove whitespaces
    outText = outText.strip()
    
    #Remove stopwords
    from nltk.corpus import stopwords 
    from nltk.tokenize import word_tokenize 
    stop_words = set(stopwords.words('english'))

    tokens = word_tokenize(outText)
    outText = [i for i in tokens if not i in stop_words]
    
    #Lemmatization
    from nltk.stem import WordNetLemmatizer
    lemmatizer=WordNetLemmatizer()
    result=[]
    for word in outText:
        result.append(lemmatizer.lemmatize(word))
        
    return result

dictionary = {}    

def createDictionary(textArr):      #creating dictionary
    for word in textArr:
        if word not in dictionary:  
            dictionary[word] = 1    #adding the word to dictionary if it is not already present
        else:
            dictionary[word] += 1   #updating the occurence number of the word

processedTrainData = {}     #stores the preprocessed version of the train data
processedTestData = {}      #stores the preprocessed version of the test data
i = 0
for document in train.data:
    result = preProcessing(document)
    createDictionary(result)    #passing the train data to the dictionary
    processedTrainData[i] = result  
    i += 1

i = 0

for document in test.data:
    result = preProcessing(document)
    createDictionary(result)    #passing the test data to the dictionary
    processedTestData[i] = result
    i += 1

import operator
sorted_dictionary = sorted(dictionary.items(), key= operator.itemgetter(1), reverse= True)  #sorting the dictionary according to the occurrence numbers

n = 10000
newDictionary = sorted_dictionary[0:n]  #forming a new dictionary with 10000 features, as stated in Q1
dictVersion = dict.fromkeys(list(zip(*newDictionary))[0], list(zip(*newDictionary))[1])     #dictionary version of newDictionary, to be used in calculateProbability function
import numpy as np
countOfTarget = len(np.unique(train.target))    #getting the number of all possible target values (20 in this case)

def fit(data, labels):
    frequencies = {}        #will store the occurence number of each word in the dictionary in different classes
    l_arr = np.array(labels)    #creating numpy array from labels array
    q5 = 0
    for target in range(countOfTarget):     #for each class/target/label
        frequencies[target] = {}        #creating a sub-dictionary data structure
        target_rows = np.where(l_arr == target)[0]  #getting the indices of the numpy array whose values are currently visited label
        d_arr = operator.itemgetter(*target_rows)(data)     #getting the documents which are classified with currently visited label
        sumOfTheWordsInTheClass = 0     #stores the total number of words in the documents which are classified with currently visited label
        for i in range(len(newDictionary)):     #for each word in the constructed di ctionary
            frequencies[target][newDictionary[i][0]] = sum(x.count(newDictionary[i][0]) for x in d_arr)     #calculating and storing the occurence number of the word in the documents which are classified with currently visited label 
            sumOfTheWordsInTheClass += frequencies[target][newDictionary[i][0]]     #updating the count of the words 
        """             #Code segment for Q5
        print("Most commonly occured words in \"",train.target_names[q5],"\" documents:")
        q5 += 1         
        list1 = sorted(frequencies[target].items(), key= operator.itemgetter(1), reverse= True)
        for i in range(10):
            print(list1[i])
        """
        frequencies[target]["COUNT"] = sumOfTheWordsInTheClass  #storing the number of words in that class under capitlized "count" to avoid collision
        frequencies[target]["PRIOR"] = len(target_rows) / len(data)
        #print("Class prior for ",target," is: ",frequencies[target]["PRIOR"]) used to print out class priors for Q2
    totalCount = 0  #stores the total number of words (that are in the dictionary) in documents from all labels
    for i in range(countOfTarget):  #for each class/target/label
        totalCount += frequencies[i]["COUNT"]   #calculating the total count
    frequencies["TOTAL_COUNT"] = totalCount  #storing the total number of words in documents from all labels under capitlized "total_count" to avoid collision
    return frequencies

def calculateProbability(frequencies, document, target):    #calculates the probability of a given document belonging to a specified label
    classPrior = np.log(frequencies[target]["PRIOR"])  #calculating the class prior 
    totalProbability = classPrior   #initially assigning total probability to class prior
    for word in document:  #for every word in the document
        occurenceNumber = 0
        if (dictVersion.get(word) != None):   #if word is one of the featues (if it's in the dictionary)
            occurenceNumber = frequencies[target][word]     #assigning the occurence number to that, otherwise it remains as 0
        word_prob = occurenceNumber + 1     #applying laplace smoothing
        total_prob = frequencies[target]["COUNT"] + len(newDictionary)  #getting the total number of words in documents that are labeled as the target, applying laplace smoothing
        wordProbability = np.log(word_prob) - np.log(total_prob)    #comuting the probability of the word appearing in documents that are classified as target
        totalProbability += wordProbability     #calculating of the total probability (addition because of the rules of logarithm)
    return totalProbability

def predict(test, frequencies):
    predictedResults = []
    for i in range(len(test)):  #for every document in the test data
        max = 0     #holds the class with the highest probability
        maxProb = float('-inf') #holds the highest probability
        for label in range(countOfTarget):  #for each class/target/label
            probOfLabel = calculateProbability(frequencies, test[i], label)  #computing the probability of the document being classified as visited label
            if (probOfLabel > maxProb):     #if the probability is greater than the known highest probability
                maxProb = probOfLabel   #updating the max probability
                max = label     #updating the class with the highest probability
        predictedResults.append(max)    #labeling the document with the class which has the highest probability
    return predictedResults

freq = fit(processedTrainData, train.target) 
labels = predict(processedTestData, freq)

from sklearn.metrics import confusion_matrix
mat = confusion_matrix(test.target, labels)
sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False, xticklabels=train.target_names, yticklabels=train.target_names)

import matplotlib.pyplot as plt
plt.xlabel("true labels")
plt.ylabel("predicted labels")
plt.show()