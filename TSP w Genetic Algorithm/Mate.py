
import Chromosome
import random

class Mate:
    def __init__(self,population,pairingType):

        #Create mothers and fathers using either
        # a. Top Down Pairing
        # b. Tournament Pairing
        if pairingType == 'topDown':
            self.fathers, self.mothers = self.topDown(population)
        elif pairingType == 'tournament':
            self.fathers, self.mothers = self.tournament(population)

        #No Offspring yet
        self.offspring = []


    def getFathers(self):
        return self.fathers

    def getMothers(self):
        return self.mothers


    # ***********************************
    # * Pairing Algorithms
    # ***********************************

    # Top Down Pairing
    def topDown(self, population):
        fathers = []
        mothers = []

        populationSize = len(population)
        i = 0
        while i < populationSize:
            fathers.append(population[i])
            mothers.append(population[i+1])
            i += 2

        return fathers, mothers


    # Tournament Pairing
    def tournament(self, population):
        
        fathers = []
        mothers = []

        # number of pairs to mate
        matingPairs = 25

        pair = 0

        while pair < matingPairs:
            subset = random.sample(population, 25)
            subset2 = random.sample(population, 25)
            parent1 = random.choice(subset)
            parent2 = random.choice(subset2)
            fathers.append(parent1)
            mothers.append(parent2)
            pair += 1

        return fathers, mothers



    # ***********************************
    # * Mating Algorithms
    # ***********************************

    #Choose which mating function to use
    def matePairs(self, matingType):
        if matingType == 'cycle':
            self.cycleCrossover()
        elif matingType == 'replacement':
            self.replacement()

        return self.offspring

    # input: two children [1:8]
    def cycleCrossover(self):
        numParents = len(self.fathers)
        for i in range(numParents):
            offspring1 = self.fathers[i].getGene()[1:8]
            offspring2 = self.mothers[i].getGene()[1:8]

            # initial swap
            temp = offspring1[0]
            toFind = offspring2[0]
            offspring1[0] = offspring2[0]
            offspring2[0] = temp
            myLength = len(offspring1)
            index = 0
            iter = 1

            while iter != index:
                if offspring1[iter] == offspring1[index]:
                    temp = offspring1[iter]
                    toFind = offspring2[iter]
                    offspring1[iter] = offspring2[iter]
                    offspring2[iter] = temp
                    index = iter

                    iter = (iter + 1) % myLength
                else:
                    iter = (iter + 1) % myLength

            offspring1 = ['a'] + offspring1 + ['a']
            offspring2 = ['a'] + offspring2 + ['a']

            self.offspring.append(Chromosome.Chromosome(offspring1))
            self.offspring.append(Chromosome.Chromosome(offspring2))


    # Single-Point Crossover with Replacement
    def replacement(self):
        numParents = len(self.fathers)
        for i in range(numParents):
            offspring1 = self.fathers[i].getGene()[1:8]
            offspring2 = self.mothers[i].getGene()[1:8]

            #Perform Single-Point Crossover
            crossoverPoint = random.randint(0,7)
            for j in range(crossoverPoint):
                swap = offspring1[j]
                offspring1[j] = offspring2[j]
                offspring2[j] = swap

            #Find Missing Values and duplicate idxs
            missing1 = ['b','c','d','e','f','g','h']
            dupIdxs1 = []
            missing2 = ['b','c','d','e','f','g','h']
            dupIdxs2 = []
            for k in range(len(offspring1)):
                if offspring1[k] in missing1:
                    missing1.remove(offspring1[k])
                else:
                    dupIdxs1.append(k)
                if offspring2[k] in missing2:
                    missing2.remove(offspring2[k])
                else:
                    dupIdxs2.append(k)

            for p in range(len(dupIdxs1)):
                offspring1[dupIdxs1[p]] = missing1[p]

            for q in range(len(dupIdxs2)):
                offspring2[dupIdxs2[q]] = missing2[q]

            offspring1 = ['a'] + offspring1 + ['a']
            offspring2 = ['a'] + offspring2 + ['a']

            self.offspring.append(Chromosome.Chromosome(offspring1))
            self.offspring.append(Chromosome.Chromosome(offspring2))






















