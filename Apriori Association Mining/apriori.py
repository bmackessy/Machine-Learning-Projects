import csv
import itertools


def k1subsets(itemset):
    result = []
    for i in range(len(itemset)):
        result.append(itemset[:i] + itemset[i + 1:])
    return result


def remove_dups(table):
    for i in range(len(table) - 1):
        for j in range(i + 1, len(table)):
            #            print table[i], table[j]
            if table[i] == table[j]:
                del table[i]
            if table[j] == table[-1]:
                break
    return


def convert(lol):
    curr = []
    for entry in lol:
        curr.append(entry[0])

    return curr


def AUB(A, B):
    AUB = []
    a = list(set(A))  # remove duplicates
    b = list(set(B))  # remove duplicates
    new = a[:]
    for element in b:
        add = True
        for entry in a:
            if element == entry:
                add = False
        if add == True:
            new.append(element)

    return sorted(new)

# To be used after smash
# give it a 2d list of strings
def grabAllVals(list):
    allVals = []
    for a in list:
        for b in a:
            if (b in allVals):
                pass
            else:
                allVals.append(b)

    return allVals

# Trims 3d arrays to 2d arrays
def smashLists(list, i):
    maxList = []
    if (i == 2):
        for a in list:
            miniList = []
            for b in a:
                miniList.append(b[0])

            maxList.append(miniList)
    else:
        for a in list:
            miniList = []
            for b in a:
                for c in b:
                    miniList.append(c)

            maxList.append(miniList)

    return maxList


def generateAssociations(table, minSupport, isTitanic):
    Lk = []
    newList = []
    for row in table:
        for entry in row:
            item = []
            if (entry in newList):
                pass
            else:
                if (getSupport(table, [entry]) >= minSupport):
                    newList.append(entry)
                    item.append(entry)
                    Lk.append(item)

    for row in Lk:
        print row

    i = 2
    while (Lk != []):
        if (i == 2):
            Lk_1 = []
            for row in Lk:
                Lk_1.append(row)

        print 'i=' + str(i)

        C = getCombinations(Lk_1, i)
        remove_dups(C)
        print 'C' + str(i)
        for row in C:
            print row

        foundOne = False
        for row in C:
            curr = convert(row[:])
            if (getSupport(table, curr) >= minSupport):
                foundOne = True
                Lk.append(row)

        print
        print 'Lk' + str(i)
        for row in Lk:
            print row


        if (not(i == 3 and isTitanic == 1)):
            Lk = smashLists(Lk, i)
            Lk_1 = grabAllVals(smashLists(table, i))


        i += 1
        print "Is Here"
        print Lk

        if (not foundOne):
            break

    return Lk


def getCombinations(association, size):
    storage = []
    pointer = []
    lastIndex = len(association) - 1
    sizePointer = size - 1

    headPointer = 0
    for i in range(sizePointer):
        pointer.append(i + 1)

    while (headPointer < (len(association) - size)):
        list = []
        list.append(association[headPointer])
        for point in pointer:
            list.append(association[point])
        storage.append(list)

        for i in range(size - 1):
            while (pointer[sizePointer - i - 1] != lastIndex - i):
                pointer[sizePointer - i - 1] += 1

                list = []
                list.append(association[headPointer])
                for point in pointer:
                    list.append(association[point])
                storage.append(list)

        headPointer += 1
        for i in range(sizePointer):
            pointer[i] = headPointer + i + 1

        list = []
        list.append(association[headPointer])
        for point in pointer:
            list.append(association[point])
        storage.append(list)

    return storage


# Both/Total
def getSupport(table, association):
    both = 0
    total = 0

    for row in table:
        itemSet = association[:]
        total += 1

        for item in row:
            if (item in itemSet):
                itemSet.pop(itemSet.index(item))
                if (itemSet == []):
                    both += 1
                    itemSet = ["Not Empty"]
                    break

        if (itemSet == []):
            both += 1

    return float(both) / float(total)


# Both/Left
def getConfidence(table, L, R):
    association = []
    for i in L:
        association.append(i)
    for i in R:
        association.append(i)

    both = 0
    left = 0

    for row in table:
        itemSet = association[:]
        leftSet = L[:]

        for item in row:
            if (item in itemSet):
                itemSet.pop(itemSet.index(item))
                if (itemSet == []):
                    both += 1
                    left += 1
                    itemSet = ["Not Empty"]
                    leftSet = ["Not Empty"]
                    break

            if (item in leftSet):
                leftSet.pop(leftSet.index(item))

        if (leftSet == []):
            left += 1
        if (itemSet == []):
            both += 1

    return float(both) / float(left)


# support(L or R) / (support(L) x support(R))
def getLift(table, L, R):
    association = []
    for i in L:
        association.append(i)
    for i in R:
        association.append(i)

    # print "support(L U R)"
    # print getSupport(table, association)
    # print "bottom"
    # print getSupport(table, L) * getSupport(table, R)

    print " "
    print "Lift:"
    return (getSupport(table, association)) / (getSupport(table, L) * getSupport(table, R))


# opens the titanic dataset
def open_dataset_titanic():
    the_file = open("titanic.txt", "r")
    the_reader = csv.reader(the_file, dialect="excel")
    table = []
    for row in the_reader:
        delete = False
        if len(row) > 0:
            for entry in row:
                if (entry == 'NA'):
                    delete = True
            if (not delete):
                table.append(row)
    return table


# opens the mushroom dataset
def open_dataset_mushrooms():
    the_file = open("agaricus-lepoita.txt", "r")
    the_reader = csv.reader(the_file, dialect="excel")
    table = []
    for row in the_reader:
        delete = False
        if len(row) > 0:
            for entry in row:
                if (entry == 'NA'):
                    delete = True
            if (not delete):
                table.append(row)
    return table


def generateRules(minConfidence, associations, table):
    allRules = []

    for association in associations:
        if (len(association) == 1):
            #print "Should have size one"
            #print association
            pass
        else:
            myPermutations = itertools.permutations(association, len(association))
            for perm in myPermutations:
                newList = []
                rightHand = []
                leftHand = []
                for i in range(len(perm)):
                    if (i == len(perm)-1):
                        rightHand.append(perm[i])
                        #print "rightHand"
                        #print rightHand
                    else:
                        leftHand.append(perm[i])
                        #print "leftHand"
                        #print leftHand
                con = getConfidence(table, leftHand, rightHand)

                if ((len(perm)-2) == 0):
                    #print "Confidence"
                    #print con
                    if (con > minConfidence):
                        newList.append(leftHand)
                        newList.append(rightHand)
                        allRules.append(newList)

                else:
                    for i in range(len(perm)-2):
                        #print "Should Never"
                        #print "Confidence"
                        #print con
                        if (con > minConfidence):
                            newList.append(leftHand)
                            newList.append(rightHand)
                            allRules.append(newList)
                            #print leftHand
                            #print rightHand
                            rightHand.append(leftHand.pop())
                        else:
                            break

    return allRules


def prettyPrint(rules):
    #for rule in rules:
        #print len(rule)

    return 0




# ____________________________________________________________________________________________________
# main

table = open_dataset_titanic()


associations = generateAssociations(table, 0.25, 1)

print generateRules(0.5, associations, table)

