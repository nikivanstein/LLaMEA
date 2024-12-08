import numpy as np
import random

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
        new_individual = self.evaluate_fitness(best_func)
        self.search_space = [x for x in self.search_space if x not in new_individual]

        return best_func

    def mutate(self, individual):
        # Refine the strategy by changing the lines of the selected solution
        # to refine its strategy
        for i in range(self.dim):
            if random.random() < 0.45:
                self.search_space[i] = random.uniform(-5.0, 5.0)
        return individual

class GeneticAlgorithm:
    def __init__(self, budget, dim):
        self.metaheuristic = Metaheuristic(budget, dim)
        self.population_size = 100
        self.population = [self.metaheuristic.__call__(func) for func in range(self.population_size)]

    def evaluate_fitness(self, individual):
        # Evaluate the fitness of each individual
        return self.metaheuristic(individual)

    def mutate_individual(self, individual):
        # Mutate the individual to refine its strategy
        return self.metaheuristic.mutate(individual)

# Test the Genetic Algorithm
ga = GeneticAlgorithm(budget=100, dim=10)
ga.population = ga.population[:10]  # Select the first 10 individuals
print(ga.evaluate_fitness(ga.population[0]))