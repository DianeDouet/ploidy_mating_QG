# The effects of ploidy and mating system on the evolvability of populations

In this repository, you will find the python file containing the code used for the simulations of the paper: "The effects of ploidy and mating system on the evolvability of populations: theoretical and empirical investigations" and the dataset used for the empirical results.

The python file contains a main function called "simulation" with different parameters defined at the end of the file: number of loci, population size, dosage, mutation rate, selfing rate, ploidy level, and number of simulations to run.
The "simulation" function calls different functions present in the same file:
- mutation: operates mutations on the genome, according to the mutation rate per haplotype
- fitness: computes the fitness using the selection function
- selection: selects the parents of the next generation
- crossover: creates the new chromosomes obtained after random permutations
- offspringGenotype, offspringPhenotype, OffspringGenome: returns the genotypes, the phenotypes and the genomes of the offspring respectively
- variance: computes the genetic variance, the genic variance and the genetic covariance
- freqAncestral: returns the frequency of an ancestral allele
- inbreedingDepression: computes inbreeding depression

