import numpy as np
import random

class BlackBoxOptimizer:
    def __init__(self, budget, dim):
        """
        Initialize the BlackBoxOptimizer with a given budget and dimensionality.

        Parameters:
        budget (int): The maximum number of function evaluations allowed.
        dim (int): The dimensionality of the search space.
        """
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        """
        Optimize the black box function `func` using the given budget and search space.

        Parameters:
        func (function): The black box function to optimize.

        Returns:
        tuple: The optimized parameters and the optimized function value.
        """
        # Initialize the population size
        population_size = 100

        # Initialize the population with random parameters
        population = np.random.uniform(-5.0, 5.0, (population_size, self.dim))

        # Evaluate the function for each individual in the population
        for _ in range(self.budget):
            # Evaluate the function for each individual in the population
            func_values = func(population)

            # Select the fittest individuals based on the function values
            fittest_individuals = np.argsort(func_values)[::-1][:self.population_size // 2]

            # Create a new population by combining the fittest individuals
            new_population = np.concatenate([population[:fittest_individuals.size // 2], fittest_individuals[fittest_individuals.size // 2:]])

            # Create a new population by refining the fittest individuals
            refined_individuals = np.concatenate([
                population[:fittest_individuals.size // 2],
                self.refine(fittest_individuals, func_values)
            ])

            # Replace the old population with the new population
            population = new_population

        # Return the optimized parameters and the optimized function value
        return population, func(population)

    def refine(self, fittest_individuals, func_values):
        """
        Refine the fittest individuals using a novel mutation strategy.

        Parameters:
        fittest_individuals (np.ndarray): The fittest individuals.
        func_values (np.ndarray): The function values.

        Returns:
        np.ndarray: The refined fittest individuals.
        """
        # Create a list of mutations
        mutations = np.random.choice([-1, 1], size=len(fittest_individuals), replace=False)

        # Apply the mutations to the fittest individuals
        refined_individuals = fittest_individuals.copy()
        for i, mutation in enumerate(mutations):
            if mutation == -1:
                # Swap two random individuals
                ref1, ref2 = random.sample(fittest_individuals, 2)
                refined_individuals[i], refined_individuals[i + 1] = ref2, ref1
            else:
                # Randomly change one random individual
                ref1, ref2 = random.sample(fittest_individuals, 2)
                if random.random() < 0.5:
                    refined_individuals[i], refined_individuals[i + 1] = ref2, ref1

        # Replace the old fittest individuals with the refined ones
        fittest_individuals = refined_individuals

        # Evaluate the function for the refined fittest individuals
        func_values = func_values[fittest_individuals]

        # Return the refined fittest individuals
        return fittest_individuals