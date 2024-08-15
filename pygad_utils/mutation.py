import pygad
import numpy as np


def random_gaussian(offspring: np.ndarray, ga_instance: pygad.GA):
    """
    Randomly modifies the offsprings with a probability controlled by `mutation_probability`.
    The mutation has a Gaussian shape with the center between `random_mutation_min_val` and `random_mutation_min_val` and sigma

    Assumes no gene space and float-value genes; does not use mutation_num_genes, mutation_by_replacement, allow_duplicate_genes.

    Parameters
    ----------
    offspring : np.ndarray
        the array of offsprings to be mutated
    ga_instance : pygad.GA
        the instance of the GA class
    """
    # generate probability multiplier for the
    probability_mult = (
        (np.random.random_sample(offspring.size) < ga_instance.mutation_probability)
        .reshape(offspring.shape)
        .astype('float')
    )

    range_min = ga_instance.random_mutation_min_val
    range_max = ga_instance.random_mutation_max_val

    range_avg = 0.5 * (range_min + range_max)
    range_sigma = range_max - range_avg

    random_change = np.random.normal(loc=range_avg, scale=range_sigma, size=offspring.size).reshape(offspring.shape)

    offspring += probability_mult * random_change

    return offspring
