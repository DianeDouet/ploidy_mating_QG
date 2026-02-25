# -*- coding: utf-8 -*-
"""
Created on Mon Jan 26 14:02:55 2026

@author: diane_d
"""

import numpy as np
import sys


#Bottleneck simulations

#Three phases of 2000 generations:
#- Diploid
#- Tetraploid: all individuals produce unreduced gametes for one generation
#- Hexaploid: 1 parent (e.g. the mother) produces unreduced gametes and the other (e.g. tha father) produces reduced gametes

#Bottlenecks:
#- from 250 to 50 individuals
#- after the bottleneck, the population gains 1 individual each generation (for 200 gen til a population of 250 individuals is reached)

#Two scenarios:
#- Rare: after 500 gen when tetraploids, then a bottleneck occurs every 500 generations
#- Frequent: directly when tetraploid, a bottleneck occurs every 250 generations

#For each scenario:
#- At least 10 replicates
#- Try the four different sets of parameters for om_2=1,9 and U=0.1,0.005

#Four figures: with both scenarios on each, and one for each set of parameters
#We look at the total genetic variance as a function of the time (or generations)
#The selfing rate is fixed at 0.95 (as the species is higly selfing)


#Mutation on the genome
def mutation(haplotype, U, L, varAddEff):
    nbMut = np.random.poisson(U) #number of mutations
    posMut = np.random.choice([i for i in range(L)],nbMut, replace = False) #positions of the mutations

    if nbMut !=0:
        for p in posMut:
            haplotype[p] = haplotype[p] + np.random.normal(0,varAddEff**0.5)

    return np.array(haplotype)


#Computes the fitness of the individuals in the population
def fitness(phenotype,opt,om_2):
    return np.exp(-((phenotype-opt)**2)/(2*om_2))

#Selects the parents for the next generation based on fitness
def selection(Npop, phenotypes, om_2, opt, selfing):
    listParents = []
    fitnessMax = np.max(fitness(phenotypes,opt,om_2))

    for _ in range(Npop):
        par1 = np.random.randint(0,len(phenotypes)) #Choosing a parent randomly in the population
        #print(par1)
        fitnessPar1 = fitness(phenotypes[par1],opt,om_2)


        while fitnessPar1/fitnessMax < np.random.rand(): #Checking that the fitness is high enough
            par1 = np.random.randint(0,len(phenotypes))
            fitnessPar1 = fitness(phenotypes[par1],opt,om_2)
        listParents.append(par1)

        if selfing <= np.random.rand(): #no selfing, i.e. outcrossing
            par2 = np.random.randint(0,len(phenotypes)) #Choosing a second parent randomly in the population
            fitnessPar2 = fitness(phenotypes[par2],opt,om_2)
            while fitnessPar2/fitnessMax < np.random.rand() or par1 == par2: #Checking that the fitness is high enough, and that the two parents are not the same
                par2 = np.random.randint(0,len(phenotypes))
                fitnessPar2 = fitness(phenotypes[par2],opt,om_2)
            listParents.append(par2)

        else: #selfing
            listParents.append(par1)

    return listParents

#Creates (ploidy) new chromosomes after random permutations
# returns a list of (ploidy) lists, corresponding to the chromosomes
def crossover(listChromosomes, L, ploidy):
    all_haplo = [[] for _ in range(ploidy)]
    for i in range(L):
        chosen_allels = np.random.permutation(ploidy)
        for h in range(ploidy):
            all_haplo[h].append(listChromosomes[chosen_allels[h]][i])
    return all_haplo

#Creates the genome of offspring
def offspringGenome(listParents, L, Npop, U, genome,ploidy, varAddEff, goTetra, goHexa):
    off_genome = []

    for n in range(Npop):
        par1, par2 = listParents[2*n], listParents[2*n+1]
        listChromosomes1 = [genome[ploidy*par1+c,:] for c in range(ploidy)]

        if goTetra == True or goHexa == True:
          haplo = crossover(listChromosomes1, L, ploidy)
        else:
          haplo = crossover(listChromosomes1, L, ploidy)[:ploidy//2]

        listChromosomes2 = [genome[ploidy*par2+c,:] for c in range(ploidy)]
        if goTetra == True:
          haplo += crossover(listChromosomes2, L, ploidy)
        else:
          haplo += crossover(listChromosomes2, L, ploidy)[:ploidy//2]
        #haplo is a list of (ploidy) chromosomes containing parental chromosomes that have been previously shuffled

        #mutation
        for i in range(len(haplo)):
            haplo[i] = mutation(haplo[i], U, L, varAddEff)
            off_genome.append(haplo[i])

    return np.array(off_genome)


#Returns the offspring phenotypes, adding a random environmental component to the genotype
def offsprintPhenotype(genotype, Npop):
    off_pheno = [0 for _ in range(Npop)]

    for i in range(Npop):
        off_pheno[i] = genotype[i]+np.random.normal(0,1)

    return np.array(off_pheno)

#Returns the offpsring genotypes, i.e. the sum of the value stored in the fitness loci
def offspringGenotype(genome, Npop, ploidy, dosage):
    off_geno = []

    for i in range(Npop):
        sum_geno = 0
        for c in range(ploidy):
            sum_geno += np.sum(genome[ploidy*i+c,:]) #sum of the value stored in the fitness loci
        off_geno.append(dosage*sum_geno)

    return np.array(off_geno)


#Computes the genetic variance
def variances(genome, genotype, Npop, L, ploidy, dosage):
    meanGenotype = np.mean(genotype)
    geneticVariance = np.sum((genotype-meanGenotype)**2)/Npop

    return geneticVariance


def freqAncestral(genome,L,Npop,ploidy):
    freqAll0 = [0 for _ in range(L)]

    for i in range(L):
        nbAll0 = 0

        for j in range(ploidy*Npop):
            if genome[j,i] == 0:
                nbAll0 += 1

        freqAll0[i] = nbAll0/(ploidy*Npop)

    return freqAll0

#Computes inbreeding depression
def inbreedingDepression(ploidy, genome, om_2, dosage, Npop,L):
    nbSample = 100 #number of parents to sample
    fitness_allof = 0
    fitness_autof = 0

    #outcrossing
    for _ in range(nbSample):
        #choosing two random parents
        par1 = np.random.randint(0,Npop)
        par2 = np.random.randint(0,Npop)
        while par1 == par2:
            par2 = np.random.randint(0,Npop)

        #creating the genome of the offspring from outcrossing
        listChromosomes1 = [genome[ploidy*par1+c,:] for c in range(ploidy)]
        haplo = crossover(listChromosomes1, L, ploidy)[:ploidy//2]

        listChromosomes2 = [genome[ploidy*par2+c,:] for c in range(ploidy)]
        haplo += crossover(listChromosomes2, L, ploidy)[:ploidy//2]

        genotype_allof = 0
        for i in range(ploidy):
            genotype_allof += np.sum(haplo[i])
        genotype_allof *= dosage
        fitness_allof += np.exp(-(genotype_allof**2)/(2*om_2)) #sum of the fitness of outcrossing individuals

    #selfing
    for _ in range(nbSample):
        #choosing one parent (selfing)
        par1 = np.random.randint(0,Npop)
        par2 = par1

        #creating the genome of the offspring from selfing
        listChromosomes1 = [genome[ploidy*par1+c,:] for c in range(ploidy)]
        haplo = crossover(listChromosomes1, L, ploidy)[:ploidy//2]

        listChromosomes2 = [genome[ploidy*par2+c,:] for c in range(ploidy)]
        haplo += crossover(listChromosomes2, L, ploidy)[:ploidy//2]

        genotype_autof = 0
        for i in range(ploidy):
            genotype_autof += np.sum(haplo[i])
        genotype_autof *= dosage
        fitness_autof += np.exp(-(genotype_autof**2)/(2*om_2)) #sum of the fitness of selfing individuals

    inbreeding_depression = 1 - fitness_autof/fitness_allof
    return inbreeding_depression


#Main program
#Returns the genetic variances at the end of each simulation.


#Scenario 1: Rare bottlenecks (after 500 gen when tetraploids, then a bottleneck occurs every 500 generations)
def simulation_1(nbSim, L, varAddEff, dosage, Npop, U, om_2, selfing, ploidy):
    varg = []
    Nmax = Npop

    for k in range(nbSim):
        print("Simulation number ",k)
        meanFitness = []
        opt = 0

        #Creating intermediary lists
        list_varg = []

        #Initialisation
        genoInit = [[0 for _ in range(L)] for _ in range(Npop*ploidy)] #initial genomes for all individuals
        genome = np.array(genoInit)
        phenotype = np.random.normal(0,1,size = Npop) #initial phenotypes of all individuals

        generation = 0
        phase = 0

        goTetra = False
        goHexa = False

        #Diploids phase
        while phase == 0:
            generation += 1
            if generation%10 == 0 or generation == 2:
              #print(generation)
              var_all = variances(genome, genotype, Npop, L, ploidy, dosage)
              list_varg.append(var_all)
            fit = fitness(phenotype,opt,om_2) #list of fitness of all individuals
            meanFitness.append(np.mean(fit)) #mean fitness of the population
            

            #selecting parents
            listParents = selection(Npop, phenotype, om_2, opt, selfing)
            #creating the genomes of the offspring
            genome = offspringGenome(listParents, L, Npop, U, genome,ploidy, varAddEff, goTetra, goHexa)
            #computing the genotypes of the offspring
            genotype = offspringGenotype(genome, Npop, ploidy, dosage)
            #computing their phenotypes
            phenotype = offsprintPhenotype(genotype, Npop)

            #Test to see if an equilibrium is reached
            if generation > 2000:
                print("Generation=", generation)
                phase += 1
                generation = 0
                goTetra = True
                print("nombre de chromosomes=",len(genome))



        #Tetraploid phase
        while phase == 1:
            dosage = 0.67

            if generation%10 == 0 and generation >0:
              #print(generation)  
              var_all = variances(genome, genotype, Npop, L, ploidy, dosage)
              list_varg.append(var_all) 

            #Bottleneck
            if Npop < Nmax:
              Npop+=1
            if generation%500==0 and generation >0:
              print("Npop=",Npop)
              Npop = 50
              print("gen bottleneck tetra=", generation)

            if generation == 1:
              goTetra = False


            generation += 1    
            fit = fitness(phenotype,opt,om_2)

            #selecting parents
            listParents = selection(Npop, phenotype, om_2, opt, selfing)
            #creating the genomes of the offspring
            genome = offspringGenome(listParents, L, Npop, U, genome,ploidy, varAddEff, goTetra, goHexa)
            #computing the genotypes of the offspring
            if generation == 1:
              ploidy = 4
            genotype = offspringGenotype(genome, Npop, ploidy, dosage)
            #computing their phenotypes
            phenotype = offsprintPhenotype(genotype, Npop)

            #Test to see if an equilibrium is reached
            if generation > 2000:
                print("Generation=", generation)
                phase += 1
                generation = 0
                goHexa = True
                print("nombre de chromosomes=",len(genome))

        #Hexaploid phase
        while phase == 2:

            if generation%10 == 0 and generation >0:
              #print(generation)
              var_all = variances(genome, genotype, Npop, L, ploidy, dosage)
              list_varg.append(var_all)
              
            #Bottleneck
            if Npop < Nmax:
              Npop+=1
            if generation%500==0 and generation > 0:
              Npop = 50
              print("gen bottleneck hexa=", generation)

            if generation == 1:
              goHexa = False


            generation += 1  
            fit = fitness(phenotype,opt,om_2)

            #selecting parents
            listParents = selection(Npop, phenotype, om_2, opt, selfing)
            #creating the genomes of the offspring
            genome = offspringGenome(listParents, L, Npop, U, genome,ploidy, varAddEff, goTetra, goHexa)
            #computing the genotypes of the offspring
            if generation == 1:
              ploidy = 6
            genotype = offspringGenotype(genome, Npop, ploidy, dosage)
            #computing their phenotypes
            phenotype = offsprintPhenotype(genotype, Npop)

            #Test to see if an equilibrium is reached
            if generation > 2000:
                print("Generation=", generation)
                phase += 1
                generation = 0
                goHexa = True
                print("nombre de chromosomes=",len(genome))



        varg.append(list_varg)
    print("Done !")

    return varg



#Scenario 2: Frequent Bottelneck (directly when tetraploid, a bottleneck occurs every 250 generations)
def simulation_2(nbSim, L, varAddEff, dosage, Npop, U, om_2, selfing, ploidy):
    varg = []
    Nmax = Npop

    for k in range(nbSim):
        print("Simulation number ",k)
        meanFitness = []
        opt = 0

        #Creating intermediary lists
        list_varg=  []

        #Initialisation
        genoInit = [[0 for _ in range(L)] for _ in range(Npop*ploidy)] #initial genomes for all individuals
        genome = np.array(genoInit)
        phenotype = np.random.normal(0,1,size = Npop) #initial phenotypes of all individuals

        generation = 0
        phase = 0

        goTetra = False
        goHexa = False

        #Diploid phase
        while phase == 0:
            generation += 1
            
            if generation%10 == 0 or generation == 2:
              #print(generation)
              var_all = variances(genome, genotype, Npop, L, ploidy, dosage)
              list_varg.append(var_all)
              
            fit = fitness(phenotype,opt,om_2) #list of fitness of all individuals
            meanFitness.append(np.mean(fit)) #mean fitness of the population

            

            #selecting parents
            listParents = selection(Npop, phenotype, om_2, opt, selfing)
            #creating the genomes of the offspring
            genome = offspringGenome(listParents, L, Npop, U, genome,ploidy, varAddEff, goTetra, goHexa)
            #computing the genotypes of the offspring
            genotype = offspringGenotype(genome, Npop, ploidy, dosage)
            #computing their phenotypes
            phenotype = offsprintPhenotype(genotype, Npop)

            #Test to see if an equilibrium is reached
            if generation > 2000:
                print("Generation=", generation)
                phase += 1
                generation = 0
                goTetra = True


        #Tetraploid phase
        while phase == 1:
            dosage = 0.67

            if generation%10 == 0 and generation >0:
              #print(generation)  
              var_all = variances(genome, genotype, Npop, L, ploidy, dosage)
              list_varg.append(var_all)

            #Bottleneck
            if Npop < Nmax:
              Npop+=1
            if generation%250==0:
              print("Npop=",Npop)
              Npop = 50
              print("gen bottleneck tetra=", generation)

            if generation == 1:
              goTetra = False


            generation += 1    
            fit = fitness(phenotype,opt,om_2)

            #selecting parents
            listParents = selection(Npop, phenotype, om_2, opt, selfing)
            #creating the genomes of the offspring
            genome = offspringGenome(listParents, L, Npop, U, genome,ploidy, varAddEff, goTetra, goHexa)
            #computing the genotypes of the offspring
            if generation == 1:
              ploidy = 4
            genotype = offspringGenotype(genome, Npop, ploidy, dosage)
            #computing their phenotypes
            phenotype = offsprintPhenotype(genotype, Npop)

            #Test to see if an equilibrium is reached
            if generation > 2000:
                print("Generation=", generation)
                phase += 1
                generation = 0
                goHexa = True

        #Hexaploid phase
        while phase == 2:

            if generation%10 == 0 and generation >0:
              #print(generation)
              var_all = variances(genome, genotype, Npop, L, ploidy, dosage)
              list_varg.append(var_all)

            #Bottleneck
            if Npop < Nmax:
              Npop+=1
            if generation%250==0 and generation > 0:
              Npop = 50
              #print("gen bottleneck hexa=", generation)

            if generation == 1:
              goHexa = False


            generation += 1  
            fit = fitness(phenotype,opt,om_2)

            #selecting parents
            listParents = selection(Npop, phenotype, om_2, opt, selfing)
            #creating the genomes of the offspring
            genome = offspringGenome(listParents, L, Npop, U, genome,ploidy, varAddEff, goTetra, goHexa)
            #computing the genotypes of the offspring
            if generation == 1:
              ploidy = 6
            genotype = offspringGenotype(genome, Npop, ploidy, dosage)
            #computing their phenotypes
            phenotype = offsprintPhenotype(genotype, Npop)

            #Test to see if an equilibrium is reached
            if generation > 2000:
                #print("Generation=", generation)
                phase += 1
                generation = 0
                goHexa = True

        varg.append(list_varg)
    print("Done !")

    return varg




#Parameters

L = 50 #number of loci
varAddEff = 0.05 #extent of the mutation
dosage = 1
Npop = 250 #population size

U = 0.005 #mutation rate
om_2 = 1 #width of the fitness function, i.e. strength of selection\ values: 1 or 9

selfing = 0.95 #selfing rate: if selfing = 1, full selfing and if selfing = 0, full outcrossing\ values: between 0 and 1
ploidy = 2 #ploidy level of all the individuals in the initial population

nbSim = 10 #number of simulations




#Running the simulation with the different parameters above
varg = simulation_1(nbSim, L, varAddEff, dosage, Npop, U, om_2, selfing, ploidy)




