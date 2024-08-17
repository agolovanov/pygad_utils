import numpy as np
import pygad


def random_gaussian(offspring: np.ndarray, ga_instance: pygad.GA) -> np.ndarray:
    """
    Randomly modifies the offsprings with a probability controlled by `mutation_probability`.
    The mutation has a Gaussian shape with the center between `random_mutation_min_val` and `random_mutation_min_val` and sigma

    Assumes no gene space and float-value genes; does not use mutation_num_genes, mutation_by_replacement, allow_duplicate_genes.

    Parameters
    ----------
    offspring : np.ndarray
        the array of offsprings to be mutated
    ga_instance : pygad.GA
        the instance of the GA class from which the randomization parameters are taken

    Returns
    -------
    np.ndarray
        2D array of mutated offspring (same shape as offspring)
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


def add_mutation_death_sentence_condition(mutation_func: callable, condition: callable) -> callable:
    """Adds a condition which does not allow mutation results to violate the constraint condition on genes.
    All genes before the mutation must satisfy the condition.

    Parameters
    ----------
    mutation_func : callable
        The mutation function.
    condition : callable
        A function which acts on the array of genes and returns an array of True or False, where True means violation of the constraint.

    Returns
    -------
    callable
        The modified mutation function
    """

    def conditional_mutation_func(offspring: np.ndarray, ga_instance: pygad.GA):
        check_offspring = condition(offspring)
        if np.any(check_offspring):
            raise ValueError('Not all offsprings initially satisfy the condition')

        check_offspring[:] = True

        initial_offspring = offspring.copy()

        while np.any(check_offspring):
            offspring[check_offspring] = mutation_func(initial_offspring[check_offspring], ga_instance)

            check_offspring = condition(offspring)
        return offspring

    return conditional_mutation_func
