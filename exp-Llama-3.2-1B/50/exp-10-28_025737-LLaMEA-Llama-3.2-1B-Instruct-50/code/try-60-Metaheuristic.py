import random
import numpy as np
import copy

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

class EvolutionaryMetaheuristic(Metaheuristic):
    def __init__(self, budget, dim):
        super().__init__(budget, dim)
        self.population_size = 100
        self.population = [copy.deepcopy(self.search_space) for _ in range(self.population_size)]

    def __call__(self, func):
        # Initialize the population
        self.population = [copy.deepcopy(self.search_space) for _ in range(self.population_size)]

        # Iterate over generations
        for generation in range(100):
            # Evaluate the function a limited number of times
            num_evals = min(self.budget, len(func(self.search_space)))

            # Select parents using tournament selection
            parents = []
            for _ in range(self.population_size):
                parent = random.choice(self.population)
                tournament = random.sample(parent, num_evals)
                tournament_values = [func(x) for x in tournament]
                tournament_index = np.argsort(tournament_values)[-1]
                parent = [x for x, y in zip(parent, tournament) if y == tournament_values[tournament_index]]
                parents.append(parent)

            # Crossover (recombination) the parents
            offspring = []
            for _ in range(self.population_size):
                parent1, parent2 = random.sample(parents, 2)
                child = [x for x in parent1 + parent2]
                if random.random() < 0.5:
                    child = random.sample(child, len(child) // 2)
                offspring.append(copy.deepcopy(child))

            # Mutate the offspring
            for individual in offspring:
                for i in range(len(individual)):
                    if random.random() < 0.05:
                        individual[i] += random.uniform(-1, 1)

            # Replace the old population with the new one
            self.population = offspring

        # Evaluate the function a limited number of times
        num_evals = min(self.budget, len(func(self.search_space)))
        func_values = [func(x) for x in random.sample(self.search_space, num_evals)]

        # Select the best function value
        best_func = max(set(func_values), key=func_values.count)

        # Update the search space
        self.search_space = [x for x in self.search_space if x not in best_func]

        return best_func

# Description: Novel Metaheuristic Algorithm for Black Box Optimization using Evolutionary Strategies
# Code: 