import numpy as np
import random

class DABU:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = np.linspace(-5.0, 5.0, dim)
        self.func_evaluations = 0
        self.population = self.initialize_population()

    def __call__(self, func):
        while self.func_evaluations < self.budget:
            func_value = func(self.search_space)
            if np.abs(func_value) < 1e-6:  # stop if the function value is close to zero
                break
            self.func_evaluations += 1
        return func_value

    def initialize_population(self):
        population = []
        for _ in range(100):  # population size is 100
            individual = [random.uniform(-5.0, 5.0) for _ in range(self.dim)]
            population.append(individual)
        return population

    def mutate(self, individual):
        mutated_individual = individual.copy()
        if random.random() < 0.5:  # mutation probability is 50%
            mutated_individual[random.randint(0, self.dim-1)] = random.uniform(-5.0, 5.0)
        return mutated_individual

    def crossover(self, parent1, parent2):
        child = parent1.copy()
        if random.random() < 0.5:  # crossover probability is 50%
            crossover_point = random.randint(0, self.dim-1)
            child[crossover_point] = parent2[crossover_point]
        return child

    def selection(self, population):
        fitnesses = [self.evaluate_function(individual) for individual in population]
        sorted_indices = np.argsort(fitnesses)
        return population[sorted_indices[:self.budget]]

    def evaluate_function(self, individual):
        func_value = self.func_evaluations(individual)
        return func_value

# Example usage:
def test_function(x):
    return np.exp(-x[0]**2 - x[1]**2)

dabu = DABU(1000, 2)  # 1000 function evaluations, 2 dimensions
print(dabu(test_function))  # prints a random value between -10 and 10

# Novel strategy: 
# 1.  Use a combination of mutation and crossover to increase exploration
# 2.  Introduce a "memory" of the best individuals to inform the selection process
dabu.memory = dabu.population[:10]  # keep the first 10 individuals as a "memory"

dabu = DABU(1000, 2)  # 1000 function evaluations, 2 dimensions
print(dabu(test_function))  # prints a random value between -10 and 10

# Description: Novel Metaheuristic Algorithm for Black Box Optimization
# Code: 