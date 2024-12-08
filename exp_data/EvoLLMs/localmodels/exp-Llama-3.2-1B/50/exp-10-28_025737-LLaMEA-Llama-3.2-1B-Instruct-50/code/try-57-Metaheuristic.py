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
        # Initialize the population with random individuals
        population = [self.__init__(self.budget, self.dim) for _ in range(100)]

        # Run the selection process
        for _ in range(10):
            # Select the fittest individuals
            fittest = sorted(population, key=lambda x: x.fitness, reverse=True)[:self.budget]

            # Select the best individual for reproduction
            parent1 = random.choice(fittest)
            parent2 = random.choice(fittest)

            # Perform crossover
            child = self.crossover(parent1, parent2)

            # Perform mutation
            child = self.mutation(child)

            # Replace the least fit individual with the new child
            population[self.budget] = child

        # Evaluate the new population
        new_population = [func(x) for x in population]

        # Select the fittest individuals
        fittest_new = sorted(new_population, key=lambda x: x.fitness, reverse=True)[:self.budget]

        # Select the best individual for replacement
        new_individual = random.choice(fittest_new)

        # Replace the least fit individual with the new individual
        population[self.budget] = new_individual

        return new_population

    def crossover(self, parent1, parent2):
        # Perform crossover with probability 0.45
        if random.random() < 0.45:
            child = parent1
            idx = random.randint(0, self.dim - 1)
            child[idx] = parent2[idx]
            return child
        else:
            return parent1

    def mutation(self, individual):
        # Perform mutation with probability 0.1
        if random.random() < 0.1:
            idx = random.randint(0, self.dim - 1)
            individual[idx] = np.random.uniform(-5.0, 5.0)
            return individual
        else:
            return individual

# Description: Novel Metaheuristic Algorithm for Black Box Optimization
# Code: 