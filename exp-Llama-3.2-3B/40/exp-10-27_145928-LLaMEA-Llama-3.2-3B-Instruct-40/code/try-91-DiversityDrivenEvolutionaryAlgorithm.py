import numpy as np
import random

class DiversityDrivenEvolutionaryAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 20
        self.crossover_probability = 0.8
        self.mutation_probability = 0.1
        self.population = self.initialize_population()

    def initialize_population(self):
        return np.random.uniform(-5.0, 5.0, (self.population_size, self.dim))

    def evaluate(self, func):
        fitnesses = func(self.population)
        self.population = self.select_parents(fitnesses)
        self.population = self.crossover(self.population)
        self.population = self.mutate(self.population)

    def select_parents(self, fitnesses):
        fitnesses = np.array(fitnesses)
        parents = np.array([self.population[np.argsort(fitnesses)[:int(self.population_size/2)]]])
        return parents

    def crossover(self, population):
        offspring = np.zeros((self.population_size, self.dim))
        for i in range(self.population_size):
            if random.random() < self.crossover_probability:
                parent1 = random.choice(population)
                parent2 = random.choice(population)
                offspring[i] = (parent1 + parent2) / 2
        return offspring

    def mutate(self, population):
        mutated_population = np.copy(population)
        for i in range(self.population_size):
            if random.random() < self.mutation_probability:
                mutated_population[i] += np.random.uniform(-1.0, 1.0, self.dim)
        return mutated_population

    def __call__(self, func):
        for _ in range(self.budget):
            self.evaluate(func)
        return np.min(self.population, axis=0)

# Example usage
def func(x):
    return np.sum(x**2)

ddea = DiversityDrivenEvolutionaryAlgorithm(budget=100, dim=10)
optimal_solution = ddea(func)
print(optimal_solution)

# Description: Novel "Differential Evolution with Diversity" algorithm that combines the benefits of differential evolution and diversity-driven evolutionary algorithm to optimize black box functions.
# Code: 
# ```python
import numpy as np
import random

class DifferentialEvolutionWithDiversity:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 20
        self.crossover_probability = 0.8
        self.mutation_probability = 0.1
        self.differential_evolution_params = {'F': 0.5, 'CR': 0.5}
        self.diversity_driven_evolution_params = {'population_size': 20, 'crossover_probability': 0.8,'mutation_probability': 0.1}
        self.population = self.initialize_population()

    def initialize_population(self):
        return np.random.uniform(-5.0, 5.0, (self.diversity_driven_evolution_params['population_size'], self.dim))

    def evaluate(self, func):
        fitnesses = func(self.population)
        self.population = self.select_parents(fitnesses)
        self.population = self.differential_evolution(self.population)
        self.population = self.diversity_driven_evolution(self.population)

    def select_parents(self, fitnesses):
        fitnesses = np.array(fitnesses)
        parents = np.array([self.population[np.argsort(fitnesses)[:int(self.diversity_driven_evolution_params['population_size']/2)]]])
        return parents

    def differential_evolution(self, population):
        offspring = np.zeros((self.population_size, self.dim))
        for i in range(self.population_size):
            rand_idx1, rand_idx2, rand_idx3 = np.random.randint(0, self.population_size, 3)
            parent1 = population[rand_idx1]
            parent2 = population[rand_idx2]
            parent3 = population[rand_idx3]
            offspring[i] = parent1 + self.differential_evolution_params['F'] * (parent2 - parent3)
        return offspring

    def diversity_driven_evolution(self, population):
        mutated_population = np.copy(population)
        for i in range(self.diversity_driven_evolution_params['population_size']):
            if random.random() < self.diversity_driven_evolution_params['mutation_probability']:
                mutated_population[i] += np.random.uniform(-1.0, 1.0, self.dim)
        return mutated_population

    def __call__(self, func):
        for _ in range(self.budget):
            self.evaluate(func)
        return np.min(self.population, axis=0)

# Example usage
def func(x):
    return np.sum(x**2)

dee_d = DifferentialEvolutionWithDiversity(budget=100, dim=10)
optimal_solution = dee_d(func)
print(optimal_solution)