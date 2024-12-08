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
    def __call__(self, func, mutation_rate):
        # Evaluate the function a limited number of times
        num_evals = min(self.budget, len(func(self.search_space)))
        func_values = [func(x) for x in random.sample(self.search_space, num_evals)]

        # Select the best function value
        best_func = max(set(func_values), key=func_values.count)

        # Update the search space
        new_individual = func(best_func)
        if random.random() < mutation_rate:
            new_individual = random.choice(self.search_space)
        self.search_space = [x for x in self.search_space if x not in new_individual]

        return best_func, new_individual

class GeneticAlgorithm(Metaheuristic):
    def __init__(self, budget, dim, population_size):
        super().__init__(budget, dim)
        self.population_size = population_size
        self.population = self.initialize_population()

    def initialize_population(self):
        # Create an initial population of random individuals
        return [[np.random.uniform(-5.0, 5.0, self.dim) for _ in range(self.dim)] for _ in range(self.population_size)]

    def fitness(self, individual):
        # Evaluate the fitness of an individual
        return self.func(individual)

    def evaluate_fitness(self, individual):
        # Evaluate the fitness of an individual
        return self.func(individual)

    def select_parents(self):
        # Select parents for the next generation
        parents = []
        for _ in range(self.population_size // 2):
            parents.append(self.select_random_parent())
        for _ in range(self.population_size // 2):
            parents.append(self.select_random_parent())
        return parents

    def select_random_parent(self):
        # Select a random parent
        return random.choice(self.population)

    def crossover(self, parent1, parent2):
        # Perform crossover between two parents
        child = [[0 for _ in range(self.dim)] for _ in range(self.dim)]
        for i in range(self.dim):
            child[i][0] = parent1[i]
            child[i][1] = parent2[i]
        return child

    def mutate(self, individual):
        # Perform mutation on an individual
        mutated_individual = individual.copy()
        if random.random() < 0.5:
            mutated_individual[0][0] = random.uniform(-5.0, 5.0)
        if random.random() < 0.5:
            mutated_individual[1][0] = random.uniform(-5.0, 5.0)
        return mutated_individual

    def __call__(self, func):
        # Evaluate the function a limited number of times
        num_evals = min(self.budget, len(func(self.search_space)))
        func_values = [func(x) for x in random.sample(self.search_space, num_evals)]

        # Select the best function value
        best_func = max(set(func_values), key=func_values.count)

        # Update the search space
        self.search_space = [x for x in self.search_space if x not in best_func]

        return best_func

# Description: Novel Metaheuristic Algorithm for Black Box Optimization
# Code: 