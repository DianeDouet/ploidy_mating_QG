import numpy as np
import sys


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
        par1 = np.random.randint(0,Npop) #Choosing a parent randomly in the population
        fitnessPar1 = fitness(phenotypes[par1],opt,om_2)
                  
        while fitnessPar1/fitnessMax < np.random.rand(): #Checking that the fitness is high enough
            par1 = np.random.randint(0,Npop)
            fitnessPar1 = fitness(phenotypes[par1],opt,om_2)
        listParents.append(par1)
                  
        if selfing <= np.random.rand(): #no selfing, i.e. outcrossing
            par2 = np.random.randint(0,Npop) #Choosing a second parent randomly in the population
            fitnessPar2 = fitness(phenotypes[par2],opt,om_2)
            while fitnessPar2/fitnessMax < np.random.rand() or par1 == par2: #Checking that the fitness is high enough, and that the two parents are not the same
                par2 = np.random.randint(0,Npop)
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
def offspringGenome(listParents, L, Npop, U, genome,ploidy, varAddEff):
    off_genome = []

    for n in range(Npop):
        par1, par2 = listParents[2*n], listParents[2*n+1]
        listChromosomes1 = [genome[ploidy*par1+c,:] for c in range(ploidy)]
        haplo = crossover(listChromosomes1, L, ploidy)[:ploidy//2]

        listChromosomes2 = [genome[ploidy*par2+c,:] for c in range(ploidy)]
        haplo += crossover(listChromosomes2, L, ploidy)[:ploidy//2]
        #haplo is a list of (ploidy) chromosomes containing parental chromosomes that have been previously shuffled
        
        #mutation
        for i in range(ploidy):
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
    if ploidy == 2:
        dosage = 1
        
    for i in range(Npop):
        sum_geno = 0
        for c in range(ploidy):
            sum_geno += np.sum(genome[ploidy*i+c,:]) #sum of the value stored in the fitness loci
        off_geno.append(dosage*sum_geno)
        
    return np.array(off_geno)


#Computes the genetic variance, the genic variance (par_locus), and the genetic covariance
def variances(genome, genotype, Npop, L, ploidy, dosage=1):
    meanGenotype = np.mean(genotype)
    geneticVariance = np.sum((genotype-meanGenotype)**2)/Npop 
    par_locus = 0 #genic variance
    
    for i  in range(L):
        phenotype_loc = [0 for _ in range(Npop)]
        
        for j in range(Npop):
            sum_geno_loc = 0
            for c in range(ploidy):
                sum_geno_loc += genome[ploidy*j+c,i] #summing the values stored in one locus on each chromosome
            phenotype_loc[j] = dosage*sum_geno_loc
            
        meanPhenotype = np.mean(phenotype_loc) 
        par_locus += np.sum((phenotype_loc - meanPhenotype)**2)/Npop #computing the additive variance
        
    geneticCov = geneticVariance - par_locus 
    summaryVariance = [geneticVariance, par_locus, geneticCov]
    
    return summaryVariance


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
    
    if ploidy == 2:
        dosage = 1
    
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
#Returns the fitness at equilibrium, the genetic and genic variances, the genetic covariance, the ancestral frequency and the inbreeding depression.

def simulation(nbSim, L, varAddEff, dosage, Npop, U, om_2, selfing, ploidy):
    fitnessEq = [] 
    varg = []
    var_add = []
    cov = []
    freq0 = []
    inDep = []
    
    if ploidy == 2:
        dosage = 1
        
    for k in range(nbSim):
        print("Simulation number ",k)
        meanFitness = []
        opt = 0
        
        #Initialisation
        genoInit = [[0 for _ in range(L)] for _ in range(Npop*ploidy)] #initial genomes for all individuals
        genome = np.array(genoInit)
        phenotype = np.random.normal(0,1,size = Npop) #initial phenotypes of all individuals
        
        generation = 0
        while True:
            generation += 1
            fit = fitness(phenotype,opt,om_2) #list of fitness of all individuals
            meanFitness.append(np.mean(fit)) #mean fitness of the population
            
            #selecting parents
            listParents = selection(Npop, phenotype, om_2, opt, selfing)
            #creating the genomes of the offspring
            genome = offspringGenome(listParents, L, Npop, U, genome,ploidy, varAddEff)
            #computing the genotypes of the offspring
            genotype = offspringGenotype(genome, Npop, ploidy, dosage)
            #computing their phenotypes
            phenotype = offsprintPhenotype(genotype, Npop)
            
            #Test to see if an equilibrium is reached
            if generation%1000 == 0 and generation > 1000:
                print("Generation=", generation)
                mean1 = np.mean(meanFitness[generation-1000:generation])
                mean2 = np.mean(meanFitness[generation-2000:generation-1000])
                if np.abs(1-mean1/mean2)<= 0.01:
                    break
        
        fitnessEq.append(np.mean(fit))
        var_all = variances(genome, genotype, Npop, L, ploidy, dosage)
        freq = freqAncestral(genome,L,Npop,ploidy)
        
        var_add.append(var_all[1])
        varg.append(var_all[0])
        cov.append(var_all[2])
        freq0.append(np.mean(freq))
        inDep.append(inbreedingDepression(ploidy, genome, om_2, dosage, Npop,L))
    print("Done !")
        
    return fitnessEq, varg, var_add, cov, freq0, inDep
        
        
            
        
#Parameters

L = 50 #number of loci
varAddEff = 0.05 #extent of the mutation
dosage = 0.67
Npop = 250 #population size
U = 0.005 #mutation rate
om_2 = 1 #width of the fitness function, i.e. strength of selection\ values: 1 or 9

selfing = 1 #selfing rate: if selfing = 1, full selfing and if selfing = 0, full outcrossing\ values: between 0 and 1
ploidy = 2 #ploidy level of all the inidividuals in the population\ values: 1,4 or 6

nbSim = 100 #number of simulations




#Running the simulation with the different parameters above
fit, varg, var_add, cov, freq0, inDep = simulation(nbSim, L, varAddEff, dosage, Npop, U, om_2, selfing, ploidy)







