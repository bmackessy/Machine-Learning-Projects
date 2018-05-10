
import TSP
import random

class Chromosome:
    def __init__(self, newGene):
        self.tsp = TSP.TSP("tsp.csv")
        self.gene = newGene
        self.cost = self.calculateCost(newGene)

    def mutate(self,mutationIntensity):
        for i in range(mutationIntensity):
            index1 = random.randint(1, 7)
            index2 = random.randint(1, 7)
            temp = self.gene[index1]
            self.gene[index1] = self.gene[index2]
            self.gene[index2] = temp

        self.cost = self.calculateCost(self.gene)


    def getGene(self):
        return self.gene

    def getCost(self):
        return self.cost

    def calculateCost(self, gene):

        currCost = 0
        for i in range(len(gene)-1):
            locA = self.gene[i]
            locB = self.gene[i+1]
            currCost += self.getTripCost(locA,locB)

        return currCost

    def getTripCost(self, locA, locB):
        matrix = self.tsp.getMatrix()
        index1 = 0
        index2 = 0

        if locA == 'a':
            index1 = 0
        elif locA == 'b':
            index1 = 1
        elif locA == 'c':
            index1 = 2
        elif locA == 'd':
            index1 = 3
        elif locA == 'e':
            index1 = 4
        elif locA == 'f':
            index1 = 5
        elif locA == 'g':
            index1 = 6
        elif locA == 'h':
            index1 = 7

        if locB == 'a':
            index2 = 0
        elif locB == 'b':
            index2 = 1
        elif locB == 'c':
            index2 = 2
        elif locB == 'd':
            index2 = 3
        elif locB == 'e':
            index2 = 4
        elif locB == 'f':
            index2 = 5
        elif locB == 'g':
            index2 = 6
        elif locB == 'h':
            index2 = 7

        return matrix[index1][index2]





