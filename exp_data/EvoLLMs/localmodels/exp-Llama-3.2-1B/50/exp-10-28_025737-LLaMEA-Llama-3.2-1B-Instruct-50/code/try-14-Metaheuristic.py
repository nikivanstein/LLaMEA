import random
import numpy as np

class Metaheuristic:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = 2 * np.random.uniform(-5.0, 5.0, (dim,))

    def __call__(self, func):
        # Evaluate the function a limited number of times
        num_evals = min(self.budget, len(func(self.search_space)))
        func_values = [func(x) for x in random.sample(self.search_space, num_evals)]

        # Select the best function value
        best_func = max(set(func_values), key=func_values.count)

        # Update the search space
        self.search_space = [x for x in self.search_space if x not in best_func]

        return best_func

class MutationMetaheuristic(Metaheuristic):
    def __call__(self, func, mutation_prob=0.1):
        # Randomly select an individual from the search space
        individual = random.sample(self.search_space, 1)[0]

        # Apply mutation to the individual
        mutated_individual = individual.copy()
        if random.random() < mutation_prob:
            mutated_individual[0] += random.uniform(-5.0, 5.0)

        # Evaluate the mutated individual
        mutated_func_values = [func(mutated_individual[i]) for i in range(self.dim)]

        # Select the best function value
        best_mutated_func = max(set(mutated_func_values), key=mutated_func_values.count)

        # Update the search space
        self.search_space = [x for x in self.search_space if x not in best_mutated_func]

        return best_mutated_func

class SelectionMetaheuristic(Metaheuristic):
    def __call__(self, func, selection_prob=0.5):
        # Evaluate the function a limited number of times
        num_evals = min(self.budget, len(func(self.search_space)))
        func_values = [func(x) for x in random.sample(self.search_space, num_evals)]

        # Select the best function value
        best_func = max(set(func_values), key=func_values.count)

        # Update the search space
        self.search_space = [x for x in self.search_space if x not in best_func]

        return best_func

class GeneticAlgorithmMetaheuristic(Metaheuristic):
    def __init__(self, budget, dim, mutation_prob=0.1, selection_prob=0.5):
        super().__init__(budget, dim)
        self.mutation_prob = mutation_prob
        self.selection_prob = selection_prob

    def __call__(self, func):
        # Initialize the population
        population = [self.__call__(func) for _ in range(100)]

        # Evaluate the population
        population = [func(individual) for individual in population]

        # Select the fittest individuals
        fittest_individuals = [individual for individual in population if individual in population[:int(population.count(fittest_individuals)/2)]]

        # Mutate the fittest individuals
        mutated_individuals = [MutationMetaheuristic(self, func)(individual, self.mutation_prob) for individual in fittest_individuals]

        # Select the fittest mutated individuals
        fittest_mutated_individuals = [individual for individual in mutated_individuals if individual in mutated_individuals[:int(mutated_individuals.count(fittest_mutated_individuals)/2)]]

        # Evaluate the fittest mutated individuals
        fittest_mutated_func_values = [func(individual) for individual in fittest_mutated_individuals]

        # Select the best function value
        best_mutated_func = max(set(fittest_mutated_func_values), key=fittest_mutated_func_values.count)

        # Update the search space
        self.search_space = [x for x in self.search_space if x not in best_mutated_func]

        return best_mutated_func

# Description: Novel Metaheuristic Algorithm for Black Box Optimization (NMABBO)
# Code: 