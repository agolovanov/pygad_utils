import numpy as np
import pygad


def add_crossover_death_sentence_condition(crossover_func: callable, condition: callable) -> callable:
    """Adds a condition which does not allow crossover results to violate the constraint condition on genes.
    All parents for the crossover must satisfy the condition.

    Parameters
    ----------
    crossover_func : callable
        The crossover function.
    condition : callable
        A function which acts on the array of genes and returns an array of True or False, where True means violation of the constraint.


    Returns
    -------
    callable
        The modified crossover function
    """

    def conditional_crossover_func(parents: np.array, offspring_size: tuple, ga_instance: pygad.GA):
        if np.any(condition(parents)):
            raise ValueError('Not all parents initially satisfy the condition')
        parents = parents.copy()

        offspring = np.zeros(offspring_size)
        to_generate = np.full(offspring_size[0], True)

        while np.any(to_generate):
            offspring[to_generate] = crossover_func(
                parents, (np.count_nonzero(to_generate), offspring_size[1]), ga_instance
            )
            to_generate = condition(offspring)
            np.random.shuffle(parents)
        return offspring

    return conditional_crossover_func


def add_crossover_death_sentence_condition_ga(ga_instance: pygad.GA, condition: callable):
    """Adds a condition to the genetic algorithm which does not allow crossover results to violate the constraint condition on genes.
    All parents for the crossover must satisfy the condition.

    Parameters
    ----------
    ga_instance : pygad.GA
        the GA to modify
    condition : callable
        A function which acts on the array of genes and returns an array of True or False, where True means violation of the constraint.
    """
    ga_instance.old_crossover = ga_instance.crossover

    ga_instance.crossover = add_crossover_death_sentence_condition(
        lambda parents, offspring_size, ga: ga.old_crossover(parents, offspring_size), condition
    )
    ga_instance.crossover_type = ga_instance.crossover
