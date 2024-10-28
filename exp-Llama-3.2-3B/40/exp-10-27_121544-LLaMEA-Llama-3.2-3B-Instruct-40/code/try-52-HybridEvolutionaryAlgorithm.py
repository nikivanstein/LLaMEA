import numpy as np
from scipy.optimize import differential_evolution

class HybridEvolutionaryAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.bounds = [(-5.0, 5.0)] * dim
        self.x0 = np.random.uniform(self.bounds[0][0], self.bounds[0][1], self.dim)

    def __call__(self, func):
        if self.budget == 0:
            return np.nan, np.nan

        def mutate(individual):
            for i in range(self.dim):
                if np.random.rand() < 0.4:
                    individual[i] += np.random.uniform(-1, 1)
                    individual[i] = max(self.bounds[i][0], min(individual[i], self.bounds[i][1]))
            return individual

        def crossover(individual1, individual2):
            child = individual1.copy()
            for i in range(self.dim):
                if np.random.rand() < 0.4:
                    child[i] = individual2[i]
            return child

        def evaluate_fitness(individual):
            return func(individual)

        def hybrid_evolution():
            population = [self.x0]
            for _ in range(self.budget):
                new_population = []
                for _ in range(len(population)):
                    individual = population.pop(0)
                    mutated_individual = mutate(individual)
                    new_population.append(mutated_individual)
                population = new_population
                population = [crossover(individual1, individual2) for individual1, individual2 in zip(population, population)]
                fitness = [evaluate_fitness(individual) for individual in population]
                best_individual = np.argmin(fitness)
                population = [population[best_individual]]
            return population[0]

        return hybrid_evolution()

# Usage
budget = 100
dim = 10
func = lambda x: x[0]**2 + x[1]**2
algorithm = HybridEvolutionaryAlgorithm(budget, dim)
best_individual = algorithm(func)
print(best_individual)