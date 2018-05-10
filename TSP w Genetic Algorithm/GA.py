
import Chromosome
import random
import Mate

class GA:
    def __init__(self):

        # ***********************************
        # * Parameters for Algorithm
        # ***********************************

        #Number of Chromosomes in each iteration
        self.populationSize = 68

        #Number of iterations to perform
        self.numIterations = 500

        #Mutation Factor for mutating (0-1.0)
        self.mutationFactor = 0.3

        #Intensity of mutation (Integer)
        self.mutationIntensity = 10

        #Pairing Types Currently Available:
        # a. topDown
        # b. tournament
        self.pairingType = 'topDown'

        #Mating Types Currently Available:
        # a. cycle
        # b. replacement
        self.matingType = 'cycle'

        # ***********************************
        # * Initial Population
        # ***********************************
        self.population = []
        for i in range(self.populationSize):
            trip = ['b','c','d','e','f','g','h']
            random.shuffle(trip)
            trip = ['a'] + trip + ['a']
            chromosome = Chromosome.Chromosome(trip)
            self.population.append(chromosome)



    def sortPopulation(self):
        self.population.sort(key=lambda x: x.cost, reverse=False)

    def printPopulationGenes(self):
        for chrom in self.population:
            print chrom.getGene()
            print "cost = " + str(chrom.getCost())

    def getPopulation(self):
        return self.population


    def evolve(self):

        self.sortPopulation() #Sort population first so we can kill off the worst half

        for i in range(self.numIterations):

            print 'Run ' + str(i+1)

            # * Kill of worst half
            self.population = self.population[:(self.populationSize/2)]

            # * Pair Population
            pairedPop = Mate.Mate(self.population,self.pairingType)

            # * Mate Pairs
            offspring = pairedPop.matePairs(self.matingType)

            # * Append Offspring to Population
            self.population += offspring

            # * Mutate New Population
            self.mutate_population()

            # * Re-sort New Population
            self.sortPopulation()

            # * Print Best Chromosome in Population
            print 'Best Trip: ' + str(self.population[0].getGene())
            print 'Cost: ' + str(self.population[0].getCost())

            # Uncomment below to see all chromosome costs at each step
            #for chrom in self.population:
             #   print chrom.getCost()

            print ''



    # ***********************************
    # * Mutation Function
    # ***********************************
    def mutate_population(self):
        for indiv in self.population:
            # limit how often an individual is mutated
            if random.random() <= self.mutationFactor:
                indiv.mutate(self.mutationIntensity)












