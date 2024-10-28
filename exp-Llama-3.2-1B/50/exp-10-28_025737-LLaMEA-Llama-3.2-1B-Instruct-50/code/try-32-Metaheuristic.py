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

class NovelMetaheuristicAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = 2 * np.random.uniform(-5.0, 5.0, (dim,))

    def __call__(self, func):
        # Initialize a new population of individuals
        new_population = self.generate_new_population()

        # Evaluate the function a limited number of times
        num_evals = min(self.budget, len(func(new_population)))

        # Select the fittest individuals
        fittest_individuals = sorted(new_population, key=lambda x: func(x), reverse=True)[:self.budget]

        # Refine the strategy by changing the best individual
        best_func = fittest_individuals[0]
        for i in range(len(fittest_individuals)):
            best_func = max(set([func(x) for x in new_population if x not in fittest_individuals[i]]), key=func)

        # Update the search space
        self.search_space = [x for x in self.search_space if x not in best_func]

        return best_func

    def generate_new_population(self):
        new_population = []
        for _ in range(self.budget):
            # Evaluate the function a limited number of times
            num_evals = min(self.budget, len(self.search_space))

            # Select the best individual
            best_individual = self.search_space[np.argmax([func(x) for x in self.search_space])]

            # Update the search space
            self.search_space = [x for x in self.search_space if x not in best_individual]

            # Add the new individual to the population
            new_population.append(best_individual)

        return new_population

# Test the algorithm
func = lambda x: np.sin(x)
algorithm = NovelMetaheuristicAlgorithm(100, 10)
best_func = algorithm(algorithm, func)
print("Best function:", best_func)