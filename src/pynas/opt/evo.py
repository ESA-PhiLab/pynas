import random
from ..core import architecture_builder
from copy import deepcopy


def gene_mutation(children, mutation_probability):
    """
    Apply mutation to a list of child chromosomes.

    Parameters:
    children (list): A list of child chromosomes.
    mutation_probability (float): The probability of a gene mutation.

    Returns:
    list: The mutated child chromosomes.
    """
    new_children = []
    for child in children:
        child = deepcopy(child)
        
        for gene_index in range(len(child.chromosome)):
            rnd = random.random()
            if rnd <= mutation_probability:
                gene = child.chromosome[gene_index]
                # Mutate based on the type of gene
                if gene[0]=='L':  # Backbone layers
                    child.chromosome[gene_index] = architecture_builder.generate_layer_code()
                elif gene[0]=='P': # Pooling layer gene
                    child.chromosome[gene_index] = architecture_builder.generate_pooling_layer_code()
                elif gene[0]=='H': # Head gene
                    break
                else:
                    raise ValueError(f"Unrecognized gene type: {gene[0]}")
        child._reparse_layers()
        new_children.append(child)
    return new_children


def single_point_crossover(parents, verbose=False):
    """
    Performs a single-point crossover between two parent chromosomes.

    Parameters:
    parents (list): The parent chromosomes as lists of genes.

    Returns:
    list: A list containing two new child chromosomes resulting from the crossover.
    The crossover point is randomly selected within the range of the shorter chromosome length.
    """

    # Determine the length of the shorter parent chromosome
    min_length = min(len(parents[0].chromosome), len(parents[1].chromosome))

    # Randomly select a crossover point, ensuring it is within the range of both chromosomes
    crossover_cutoff = random.randint(1, min_length - 2)
    
    if verbose:
        print(f"Cut off: {crossover_cutoff}")
        print(f"Parent 0 chromosome: {parents[0].chromosome}")
        print(f"Parent 1 chromosome: {parents[1].chromosome}")

    # Perform crossover
    children = parents.copy()
    children[0].chromosome = parents[0].chromosome[:crossover_cutoff] + parents[1].chromosome[crossover_cutoff:]
    children[1].chromosome = parents[1].chromosome[:crossover_cutoff] + parents[0].chromosome[crossover_cutoff:]

    if verbose:
        print("Crossed over.")
        print(f"Child 0 chromosome: {children[0].chromosome}")
        print(f"Child 1 chromosome: {children[1].chromosome}")

    for child in children:
        child._reparse_layers()    
    
    return children
