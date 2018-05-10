import csv
import numpy
from math import log
from random import randint, shuffle
from tabulate import tabulate

# Return a subset of size F
def randomAttributeSubset(att_indexes, F):
    subset = []
    for item in att_indexes:
        subset.append(item)
    while (numLeft(subset) > F):
        subset[randint(0, len(subset) - 1)] = None
    return subset


def numLeft(list):
    total = 0
    for item in list:
        if item != None:
            total += 1
    return total


# partitions dataset(random and stratified without replacement)
# returns a list of two tables
# the first table is the remainder set
# the second table is the test set
def createTestRemainderSets(processedData, class_domain, class_index):
    newData = processedData[:]
    partitions = []
    sets = []
    testSet = []
    remainderSet = []

    for i in class_domain:
        table = []
        for row in newData:
            if (row[class_index] == i):
                table.append(row)

        partitions.append(table)

    for i in partitions:  # i is a 2-d list, each list only has one class
        shuffle(i)
        for j in range(len(i) / 3):  # j is an integer
            testSet.append(i[j])
            i.pop(j)
        for k in i:  # k is an instance
            remainderSet.append(k)

    sets.append(remainderSet)
    sets.append(testSet)

    return sets


# splits the remainder set into training and test sets using random sampling with replacement
# returns a list with two tables
# the first table contains the training set
# the second table contains the test set
def createTrainingTestSet(remainderSet):
    newSet = remainderSet[:]
    sets = []
    trainingSet = []
    testSet = []
    isSelected = []
    size = len(newSet)

    for i in range(size):
        isSelected.append(0)

    for i in range(size):
        num = randint(0, size - 1)
        trainingSet.append(newSet[num])
        if (isSelected[num] == 0):
            isSelected[num] = 1

    for i in range(size):
        if (isSelected[i] == 0):
            testSet.append(newSet[i])

    sets.append(trainingSet)
    sets.append(testSet)

    return sets


# Calculates E_new for each attribute, then returns the attribute with the lowest E_new.
def selectAttributeEnsemble(instances, attIndexes, F, classIndex):
    array = []

    if (numLeft > F):
        temp_att_indexes = randomAttributeSubset(attIndexes, F)
    else:
        temp_att_indexes = attIndexes[:]

    for i in temp_att_indexes:
        if (i == None):
            array.append(100)
        else:
            array.append(calcEnew(instances, i, classIndex))

    for att in attIndexes:
        if (array.index(min(array)) == att):
            return att

# Generates a random Forest
def genRandomForest(N, M, F, processedData, attIndexes, numBins, classIndex, parentInstance):
    entire_forest = []
    # make N trees
    for i in range(N):
        sets = createTrainingTestSet(processedData)
        trainingSet = sets[0]
        testSet = sets[1]
        decision_tree = tdidtEnsemble(trainingSet, attIndexes[:], classIndex,
                                    numBins, classDomain, parentInstance, classIndex, F)  # titanic function

        accuracy = getAccuracy(decision_tree, testSet, classIndex)
        entire_forest.append([decision_tree, accuracy])

    # take M best trees from N
    best_forest = []
    for j in range(M):
        curr_best = 0
        for curr in range(len(entire_forest)):
            if float(entire_forest[curr][1]) > float(entire_forest[curr_best][1]):
                curr_best = curr
        best_forest.append(entire_forest.pop(curr_best))

    return best_forest


# Constructs a decision tree using the TDIDT algorithm
# Returns the root node
def tdidtEnsemble(instances, attIndexes, classIndex, numBins, classDomain, minNum, parentInstances, F):
    if (len(instances) == 0):
        return Label(resolveClashes(parentInstances, classIndex, classDomain))

    # If the partition only has one unique class label
    if (sameClass(instances, classIndex) == 1):
        if (len(instances[0]) == 1):
            return Label(instances[classIndex])  # If the partition contains one instance
        else:
            return Label(instances[0][classIndex])  # there must be at least one instance, all same class label

    # Checks for >66% majority, places a label node is there is a supermajority with: data = val with supermajority
    if (superMajority(instances, classIndex, classDomain) < 408):  # 408 is the code for when there is not a supermajority
            return Label(superMajority(instances, classIndex, classDomain))

    # Checks to see if any attributes still remain
    done = True
    for entry in attIndexes:
        if (entry != None):
            done = False

    if (done == True):
        return Label(resolveClashes(instances, classIndex, classDomain))

    # Checks if there are more than x instances
    # returns most common label if the partition contains less than x isntances
    # if (len(instances) < minNum):
    #    return Label(resolveClashes(instances, classIndex, classDomain))

    else:
        newAtt = selectAttributeEnsemble(instances, attIndexes, F, classIndex)
        attIndexes[newAtt] = None

        newNode = attribute(numBins)
        newNode.type = str(newAtt)
        partitions = partitionInstances(instances, newAtt, numBins)
        for i in range(numBins):
            newNode.bins[i] = tdidtEnsemble(partitions[i], attIndexes, classIndex, numBins, classDomain, minNum,
                                            instances, F)

        return newNode


# Given a test set and the root of a tree
# Returns accuracy (right/total)
def getAccuracy(root, testSet, classIndex):
    right = 0
    total = 0

    for row in testSet:
        total += 1
        if (row[classIndex] == tdidtClassifier(root, row)):
            right += 1

    return float(right) / float(total)


# Creates a confusion matrix for dataset.
def init_confusion_matrix(classDomain):
    confusion = []
    first = True
    second = False
    for x in range(1,4):
        if (first):
            toAdd = ["1"]
            first = False
            second = True
        elif (second == True):
            toAdd = ["2"]
            second = False
        else:
            toAdd = ["3"]

        for i in range(len(classDomain)):
            toAdd.append(0)

        confusion.append(toAdd)
    return confusion

# Return the vote of the random forest
def get_label_maj_voting(forest, toPredict):
    labels = []

    for tree in forest:
        labels.append(tdidtClassifier(tree[0], toPredict))

    label = max(set(labels), key=labels.count)  # get the majority
    return label

# Generate confusion matrix data for the random forest
def method4RandomForest(table, N, M, F, classIndex, classDomain, attIndexes, numBins, parentInstance):
    # init confusion
    confusion = init_confusion_matrix(classDomain)  # confusion matric
    sets = createTestRemainderSets(table, classDomain, classIndex)

    remainderSet = sets[0]
    testSet = sets[1]

    forest = genRandomForest(N, M, F, remainderSet, attIndexes[:], numBins, classIndex, parentInstance)

    TP = 0
    for row in testSet:
        toPredict = []
        actual = row[classIndex]
        for att in attIndexes:
            toPredict.append(row[att])

        for label in classDomain:
            if (actual == label):
                actual = int(label)

        predicted = get_label_maj_voting(forest, toPredict)

        for label in classDomain:
            if (predicted == label):
                predicted = int(label)

        if (int(predicted) == 0):
            predicted = "1"
        confusion[int(actual) - 1][int(predicted)] += 1
        for i in classDomain:
            confusion[0][int(i)-1] = int(i)
        if str(actual) == str(predicted):
            TP += 1
    accuracy = TP / float(len(testSet))
    error = 1 - accuracy
    accuracy = "%.2f" % accuracy  # round to two decimal places
    error = "%.2f" % error  # round to two decimal places

    print 'Random Forest: '
    print '    Accuracy:', accuracy, 'Error rate:', error

    # fill in 'total' and 'recognition' columns for confusion matrix
    for row in confusion:
        sum = 0
        for entry in row[1:]:  # dont add first entry in row, that is the ranking
            sum += int(entry)
        row.append(sum)
        if (sum != 0):
            recognition = 1
            # row[row[0]] is where actualIndex == predictedIndex

            temp = int(row[0])
            recognition = row[temp] / float(sum) * 100
            recognition = "%.2f" % recognition  # 2 dec places
        else:
            recognition = 0
        row.append(recognition)

        # print confusion
    print '==========================================='
    print 'Confusion Matrix'
    print '==========================================='

    print tabulate(confusion, headers=['Grossing', '1', '2', '3', 'Total', 'Recognition (%)'])

    print
    print
    print
    return



# Classifies an instance given a decision tree
def tdidtClassifier(node, instance):
    if (node.type == "Label"):
        label = node.data

    else:
        currentAtt = int(node.type)
        nextDirection = int(instance[currentAtt])
        label = tdidtClassifier(node.bins[nextDirection], instance)

    return label


# Given a list of counts, returns the index of the largest value
# classCounts = partitionStats
def resolveClashes(partition, classIndex, classDomain):
    classCounts = [0] * len(classDomain)

    for row in partition:
        classCounts[int(row[classIndex]) - 1] += 1

    label = str(classCounts.index(max(classCounts)) + 1)

    return label


class attribute(object):
    def __init__(self, numBins):
        self.bins = [None] * numBins
        self.type = "TBD"


class Label():
    def __init__(self, label):
        self.data = label
        self.type = "Label"

# Returns the the index of the attribute that has a super majority
# If no attribute has a super majority, then returns: 408
def superMajority(partition, classIndex, classDomain):
    classCounts = [0] * len(classDomain)

    for row in partition:
        classCounts[int(row[classIndex]) - 1] += 1

    for i in range(len(classCounts)):
        if ((classCounts[i] / len(partition)) >= 0.66):
            return i

    return 408                      # Code for no super majority


# Constructs a decision tree using the TDIDT algorithm
# Returns the root node
def tdidt(instances, attIndexes, classIndex, numBins, classDomain, minNum, parentInstances):
    if (len(instances) == 0):
        return Label(resolveClashes(parentInstances, classIndex, classDomain))

    # If the partition only has one unique class label
    if (sameClass(instances, classIndex) == 1):
        if (len(instances[0]) == 1):
            return Label(instances[classIndex])     # If the partition contains one instance
        else:
            return Label(instances[0][classIndex])  # there must be at least one instance, all same class label

    # Checks for >66% majority, places a label node is there is a supermajority with: data = val with supermajority
    if (superMajority(instances, classIndex, classDomain) < 408):   #408 is the code for when there is not a supermajority
        return Label(superMajority(instances, classIndex, classDomain))

    # Checks to see if any attributes still remain
    done = True
    for entry in attIndexes:
        if (entry != None):
            done = False

    if (done == True):
        return Label(resolveClashes(instances, classIndex, classDomain))

    # Checks if there are more than x instances
    # returns most common label if the partition contains less than x isntances
    #if (len(instances) < minNum):
    #    return Label(resolveClashes(instances, classIndex, classDomain))

    else:
        newAtt = selectAttribute(instances, attIndexes, classIndex)
        attIndexes[newAtt] = None

        newNode = attribute(numBins)
        newNode.type = str(newAtt)
        partitions = partitionInstances(instances, newAtt, numBins)

        for i in range(numBins):
            newNode.bins[i] = tdidt(partitions[i], attIndexes, classIndex, numBins, classDomain, minNum, instances)

        return newNode


def calcEnew(instances, att_index, class_index):
    # get the length of the partition
    D = len(instances)
    # calculate the partition stats for att_index (see below)
    freqs = attributeFrequencies(instances, att_index, class_index)
    # find E_new from freqs (calc weighted avg)
    E_new = 0
    for att_val in freqs:
        D_j = float(freqs[att_val][1])
        probs = [(c / D_j) for (_, c) in freqs[att_val][0].items()]
        for p in probs:
            if (p == 0.0):
                probs[probs.index(p)] = 1  # this will keep log function from getting 0, but still having next line +0
        E_D_j = -sum([p * log(p, 2) for p in probs])
        E_new += (D_j / D) * E_D_j

    return E_new


# Returns the class frequencies for each attribute value:
# {att_val:[{class1: freq, class2: freq, ...}, total], ...}
def attributeFrequencies(instances, att_index, class_index):
    # get unique list of attribute and class values
    att_vals = list(set(getColumn(instances, att_index)))
    class_vals = list(set(getColumn(instances, class_index)))
    # initialize the result
    result = {v: [{c: 0 for c in class_vals}, 0] for v in att_vals}
    # build up the frequencies
    for row in instances:
        label = row[class_index]
        att_val = row[att_index]
        result[att_val][0][label] += 1
        result[att_val][1] += 1
    return result


# returns index of selected attribute
def selectAttribute(instances, attIndexes, classIndex):
    size = len(attIndexes)
    array = []

    for i in attIndexes:                    # Makes sure already selected attributes are given default values
        if (i == None):                     # Keeps the size of the array, the same as the total attribute size
            array.append(100)
        else:
            array.append(calcEnew(instances, i, classIndex))

    return array.index(min(array))


# Record the counts for each class label in a given partition
# Returns an array of the counts, whose index correspond with their in index in classDomain
def partitionStats(instances, classIndex, classDomain):
    totalCount = 0
    counts = []
    for i in classDomain:
        counts.append(0)

    for row in instances:
        totalCount += 1
        for i in range(len(classDomain)):
            if (int(row[classIndex]) == classDomain[i]):

                counts[i] += 1

    return counts


# Return 0 if there are muliple classes in the partition
# Return 1 if there is one class in the partition
def sameClass(instances, classIndex):
    if (len(instances[0]) == 1):
        return 1

    firstLabel = instances[0][classIndex]

    for row in instances:
        if (row[classIndex] == firstLabel):
            return 0

    return 1

# partitions instances based on attribute index
def partitionInstances(instances, index, numBins):
    table = [None] * numBins
    for i in range(numBins):
        list = []
        for row in instances:
            if (int(row[index]) == i):
                list.append(row)

        table[i] = list

    return table

# Takes in probabilites and returns the largest one
def naiveBayes(instance, table, classDomain, attIndexes, classIndex):
    probs = []
    for label in classDomain:
        probs.append(probCgivenX(instance, table, attIndexes, classIndex, label))

    return classDomain[probs.index(max(probs))]

# Calculates the conditional probability for a classifier
def probCgivenX(instance, table, attIndexes, classIndex, currentClassifier):
    total = 0
    c = 0
    counts = []

    for att in attIndexes:
        counts.append(0)

    for row in table:
        total += 1
        if (row[classIndex] == currentClassifier):
            c += 1
            for i in attIndexes:

                if (instance[i] == row[i]):
                    counts[i] += 1

    c = float(c)
    total = float(total)

    cDivTotal = c/total
    prob = 1.0
    for val in counts:
        prob = prob * val / c

    prob = prob * cDivTotal


    return prob


# the euclidean distance between row and instance, using inputIndexes
def distance(row, instance, attIndexes):
    sum = 0
    for index in attIndexes:
        sum += (float(row[index]) - float(instance[index]))**2
    dist = (sum)
    return dist


# Given an instance and the table, the function finds the 5 nearest neighbors and there class labels
# Most common class label is returned
def kNNClassifier(table, attIndexes, classIndex, instance, k):
    distances = []
    labels = []
    for i in range(k):
        distances.append(2)       #Is greater than sqrt(2) (largest possible euclidean distance for normalized data)
        labels.append("")

    for row in table:
        myVar = distance(row, instance, attIndexes)
        if (myVar == 0):
            pass
        elif (myVar < max(distances)):
            labels[distances.index(max(distances))] = row[classIndex]
            distances[distances.index(max(distances))] = myVar

    #print distances
    #print labels
    return max(set(labels), key=labels.count)


# input a table, indexes for attributes, and classIndex
# each attribute is put into one of six bins
# the max and min of each bin were decided based on stdev and mean data
# returns table with only categorical attributes
def categoricalBinningVariance(table, attIndexes, classIndex):
    stdev = []
    mean = []
    # Calculate stdev and mean for each column
    for i in attIndexes:
        floatColumn = []
        column = getColumn(table, i)
        for item in column:
             floatColumn.append(float(item))          # Cast all items to float

        stdev.append(numpy.std(floatColumn))
        mean.append(numpy.mean(floatColumn))

    # Constructs table of discretized float values, with a class label
    newTable = []
    for row in table:
        list = []
        for i in attIndexes:
            if (float(row[i]) < (mean[i] - stdev[i])):
                list.append("0")

            elif (float(row[i]) < (mean[i] - 0.5*stdev[i])):
                list.append("1")

            elif (float(row[i]) < mean[i]):
                list.append("2")

            elif (float(row[i]) < (mean[i] + 1*stdev[i])):
                list.append("3")

            else:
                list.append("4")



        list.append(row[classIndex])
        newTable.append(list)

    return newTable

# Discretizes continuous class values to 1, 2, or 3
def discretizeClass(classVal):
    if (float(classVal) < 10000000.0):
        val = "1"
    elif (float(classVal) < 50000000.0):
        val = "2"
    else:
        val = "3"

    return val


# input a table, indexes for attributes
# smallest value ~ 0, largest value  ~ 1
# normalization for k-nearest neighbors
def normalizeAtributes(table, attIndexes, classIndex):
    minList = []                        # Keeps track of min values for each attribute
    maxList = []                        # Keeps track of max values for each attribute

    # Calculate min and max for each column
    for i in attIndexes:
        floatColumn = []
        column = getColumn(table, i)
        for item in column:
             floatColumn.append(float(item))          # Cast all items to float

        attMin = min(floatColumn)
        attMax = max(floatColumn)
        minList.append(attMin)
        maxList.append(attMax)

    # Constructs table of normalized float values, with a class label
    newTable = []
    for row in table:
        list = []
        for i in attIndexes:
            list.append((float(row[i]) - minList[i])/(maxList[i]-minList[i]))      #normalize
        list.append(row[classIndex])
        newTable.append(list)

    return newTable


# returns a list containing the values in a column
def getColumn(table, columnIndex):
    list = []
    for row in table:
        list.append(row[columnIndex])

    return list

# removes empty and incorrect values (0)
# returns complete table
def cleanData(table):
    newTable = []

    for row in table:
        delete = False
        for entry in row:
            if (entry.strip() == ""):
                delete = True

        if (not delete):
            newTable.append(row)

    return newTable

# pass a table and desired attribute indexes
# removes columns of unused attributes
# discretizes the class into equal sized bins
# returns a new table with indexes for attributes
def processData(table, originalIndexes, classIndex):
    newTable = []
    for row in table:
        list = []
        for i in originalIndexes:
            list.append(row[i])
        newTable.append(list)

    for row in newTable:
        row[classIndex] = discretizeClass(row[classIndex])

    return newTable

# Tests K-nearest
def testAccuracyKNearest(trainingSet, testSet, classIndex, k):
    TP = 0
    total =  0
    for row in testSet:
        total += 1
        myLabel = kNNClassifier(trainingSet, attIndexes, classIndex, row, k)
        actualLabel = row[classIndex]
        if (myLabel == actualLabel):
            TP += 1

    return float(TP)/float(total)

# tests naive-bayes
def testAccuracyNaiveBayes(trainingSet, testSet, classIndex, classDomain, attIndexes):
    TP = 0
    total =  0
    count = 0
    for row in testSet:

        total += 1
        myLabel = naiveBayes(row, trainingSet, classDomain, attIndexes, classIndex)
        actualLabel = row[classIndex]
        if (myLabel == actualLabel):
            TP += 1

    return float(TP)/float(total)

# Tests decision tree
def testAccuracyDecisionTree(trainingSet, testSet, classIndex, classDomain, attIndexes, numBins, minNumInstances,
                             holdsParentsPartition):
    sum = 0
    TP = 0
    total =  0
    count = 0
    root = tdidt(trainingSet, attIndexes[:], classIndex, numBins, classDomain, minNumInstances,
                 holdsParentsPartition)
    for row in testSet:
        #if (count > 500):
        #    break

        total += 1
        myLabel = tdidtClassifier(root, row)
        actualLabel = row[classIndex]
        #print "Predicted: " + str(myLabel) + "   Actual: " + actualLabel
        if (myLabel == actualLabel):
            TP += 1
        else:
            sum = sum + count
        count += 1

    return float(TP)/float(total)

# Generate confusion matrix data for decision tree
def method4DecisionTree(table, classIndex, classDomain, attIndexes, numBins, minNumInstances):
    # init confusion
    confusion = init_confusion_matrix(classDomain)  # confusion matric
    sets = createTrainingTestSet(table)

    trainingSet = sets[0]
    testSet = sets[1]

    root = tdidt(trainingSet, attIndexes[:], classIndex, numBins, classDomain, minNumInstances, [])
    TP = 0
    for row in testSet:
        toPredict = []
        actual = row[classIndex]
        for att in attIndexes:
            toPredict.append(row[att])

        for label in classDomain:
            if (actual == label):
                actual = int(label)

        predicted = tdidtClassifier(root, row)

        for label in classDomain:
            if (predicted == label):
                predicted = int(label)

        if (int(predicted) == 0):
            predicted = "1"
        confusion[int(actual) - 1][int(predicted)] += 1
        for i in classDomain:
            confusion[0][int(i)-1] = int(i)
        if str(actual) == str(predicted):
            TP += 1
    accuracy = TP / float(len(testSet))
    error = 1 - accuracy
    accuracy = "%.2f" % accuracy  # round to two decimal places
    error = "%.2f" % error  # round to two decimal places

    print 'Decision Tree: '
    print '    Accuracy:', accuracy, 'Error rate:', error

    # fill in 'total' and 'recognition' columns for confusion matrix
    for row in confusion:
        sum = 0
        for entry in row[1:]:  # dont add first entry in row, that is the ranking
            sum += int(entry)
        row.append(sum)
        if (sum != 0):
            recognition = 1
            # row[row[0]] is where actualIndex == predictedIndex

            temp = int(row[0])
            recognition = row[temp] / float(sum) * 100
            recognition = "%.2f" % recognition  # 2 dec places
        else:
            recognition = 0
        row.append(recognition)

        # print confusion
    print '==========================================='
    print 'Confusion Matrix'
    print '==========================================='

    print tabulate(confusion, headers=['Grossing', '1', '2', '3', 'Total', 'Recognition (%)'])

    print
    print
    print
    return

# Generate confusion matrix data for naive-bayes
def method4NaiveBayes(table, classIndex, classDomain, attIndexes):
    # init confusion
    confusion = init_confusion_matrix(classDomain)  # confusion matric
    sets = createTrainingTestSet(table)

    trainingSet = sets[0]
    testSet = sets[1]

    TP = 0
    for row in testSet:
        toPredict = []
        actual = row[classIndex]
        for att in attIndexes:
            toPredict.append(row[att])

        for label in classDomain:
            if (actual == label):
                actual = int(label)

        predicted = naiveBayes(row, trainingSet, classDomain, attIndexes, classIndex)

        for label in classDomain:
            if (predicted == label):
                predicted = int(label)

        if (int(predicted) == 0):
            predicted = "1"
        confusion[int(actual) - 1][int(predicted)] += 1
        for i in classDomain:
            confusion[0][int(i)-1] = int(i)
        if str(actual) == str(predicted):
            TP += 1
    accuracy = TP / float(len(testSet))
    error = 1 - accuracy
    accuracy = "%.2f" % accuracy  # round to two decimal places
    error = "%.2f" % error  # round to two decimal places

    print 'Naive-Bayes: '
    print '    Accuracy:', accuracy, 'Error rate:', error

    # fill in 'total' and 'recognition' columns for confusion matrix
    for row in confusion:
        sum = 0
        for entry in row[1:]:  # dont add first entry in row, that is the ranking
            sum += int(entry)
        row.append(sum)
        if (sum != 0):
            recognition = 1
            # row[row[0]] is where actualIndex == predictedIndex

            temp = int(row[0])
            recognition = row[temp] / float(sum) * 100
            recognition = "%.2f" % recognition  # 2 dec places
        else:
            recognition = 0
        row.append(recognition)

        # print confusion
    print '==========================================='
    print 'Confusion Matrix'
    print '==========================================='

    print tabulate(confusion, headers=['Grossing', '1', '2', '3', 'Total', 'Recognition (%)'])

    print
    print
    print
    return

# Generate confusion matrix data for k-nearest neighbors
def method4KNearest(table, classIndex, k):
    # init confusion
    confusion = init_confusion_matrix(classDomain)  # confusion matrix
    sets = createTrainingTestSet(table)
    counter = 0
    trainingSet = sets[0]
    testSet = sets[1]
    print "K-Nearest is quite slow (takes a minute or so)"
    TP = 0
    for row in testSet:
        counter += 1
        if (counter == 300):
            print "33% Done"
        if (counter == 600):
            print "66% Done"
        if (counter == 900):
            print "99% Done"

        toPredict = []
        actual = row[classIndex]
        for att in attIndexes:
            toPredict.append(row[att])

        for label in classDomain:
            if (actual == label):
                actual = int(label)

        predicted = kNNClassifier(kNearestTable, attIndexes, classIndex, row, k)

        for label in classDomain:
            if (predicted == label):
                predicted = int(label)

        if (int(predicted) == 0):
            predicted = "1"
        confusion[int(actual) - 1][int(predicted)] += 1
        for i in classDomain:
            confusion[0][int(i)-1] = int(i)
        if str(actual) == str(predicted):
            TP += 1
    accuracy = TP / float(len(testSet))
    error = 1 - accuracy
    accuracy = "%.2f" % accuracy  # round to two decimal places
    error = "%.2f" % error  # round to two decimal places

    print 'K-Nearest: '
    print '    Accuracy:', accuracy, 'Error rate:', error

    # fill in 'total' and 'recognition' columns for confusion matrix
    for row in confusion:
        sum = 0
        for entry in row[1:]:  # dont add first entry in row, that is the ranking
            sum += int(entry)
        row.append(sum)
        if (sum != 0):
            recognition = 1
            # row[row[0]] is where actualIndex == predictedIndex

            temp = int(row[0])
            recognition = row[temp] / float(sum) * 100
            recognition = "%.2f" % recognition  # 2 dec places
        else:
            recognition = 0
        row.append(recognition)

        # print confusion
    print '==========================================='
    print 'Confusion Matrix'
    print '==========================================='

    print tabulate(confusion, headers=['Grossing', '1', '2', '3', 'Total', 'Recognition (%)'])

    print
    print
    print
    return


#***********************************************************************************************************************
# main function

the_file = open("imdb.txt", "r")                        # opens file
the_reader = csv.reader(the_file, dialect="excel")      # reads as csv
table = []

for row in the_reader:                                  # create table of data
	if len(row) > 0:
		table.append(row)


originalAttIndexes = [13, 2, 12, 25, 8, 23, 22]
attIndexes = [0, 1, 2, 3, 4, 5]
classIndex = 6
classDomain = ["1", "2", "3"]

# cleans and processes data
table = processData(cleanData(table), originalAttIndexes, classIndex)


# For testing k_NN
kNearestTable = normalizeAtributes(table, attIndexes, classIndex)
KnTrainingSet = createTrainingTestSet(kNearestTable)[0]
KnTestSet = createTrainingTestSet(kNearestTable)[1]
k = 5

categoricalTable = categoricalBinningVariance(table, attIndexes, classIndex)
CatTrainingSet = createTrainingTestSet(categoricalTable)[0]
CatTestSet = createTrainingTestSet(categoricalTable)[1]
numBins = 6
minNumInstances = 3
holdsParentsPartition = []

# Ensemble Method
N = 5
M = 3
F = 7

method4DecisionTree(categoricalTable, classIndex, classDomain, attIndexes[:], numBins, minNumInstances)
method4RandomForest(categoricalTable, N, M, F, classIndex, classDomain, attIndexes[:], numBins, holdsParentsPartition)
method4NaiveBayes(categoricalTable, classIndex, classDomain, attIndexes)
method4KNearest(kNearestTable, classIndex, k)