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

class NMABA(Metaheuristic):
    def __init__(self, budget, dim, p1=0.5, p2=0.5):
        super().__init__(budget, dim)
        self.p1 = p1
        self.p2 = p2

    def __call__(self, func):
        # Initialize the population with random individuals
        population = [self.__call__(func) for _ in range(100)]

        # Evaluate the population and select the fittest individuals
        fitnesses = [self.__call__(func) for func in population]
        idx = np.argsort(fitnesses)[-self.budget:]
        population = [population[i] for i in idx]

        # Perform crossover and mutation
        children = []
        while len(population) > 0:
            parent1, parent2 = random.sample(population, 2)
            child = (self.p1 * parent1 + self.p2 * parent2) / 2
            children.append(child)
            population.remove(child)

        # Evaluate the children and select the fittest ones
        fitnesses = [self.__call__(func) for func in children]
        idx = np.argsort(fitnesses)[-self.budget:]
        children = [children[i] for i in idx]

        # Mutate the children
        for child in children:
            if random.random() < 0.01:
                child = random.uniform(-1, 1)
                child = [x + child for x in child]

        # Update the population
        population = children

        return population

# Description: Novel Metaheuristic Algorithm for Black Box Optimization (NMABA)
# Code: 