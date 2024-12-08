import numpy as np
from scipy.optimize import differential_evolution

class HybridEvolutionaryAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = (-5.0, 5.0)
        self.elitism_ratio = 0.2
        self.probability = 0.05

    def __call__(self, func):
        population = np.random.uniform(self.search_space[0], self.search_space[1], size=(self.budget, self.dim))
        elite_set = population[:int(self.budget * self.elitism_ratio)]

        for _ in range(self.budget - len(elite_set)):
            fitness = np.array([func(x) for x in population])

            new_population = []
            for _ in range(len(elite_set)):
                parent1, parent2 = np.random.choice(population, size=2, replace=False)
                child = self.crossover(parent1, parent2)
                child = self.mutate(child)
                new_population.append(child)

            new_population = np.array(new_population)
            new_population = np.array([self.evaluate_fitness(func, x) for x in new_population])

            population = np.concatenate((elite_set, new_population[:len(elite_set)]))

        best_solution = np.min(func(population))
        return best_solution

    def crossover(self, parent1, parent2):
        if np.random.rand() < self.probability:
            child = 0.5 * (parent1 + parent2)
        else:
            child = parent1
        return child

    def mutate(self, individual):
        mutation = np.random.uniform(-1.0, 1.0, size=self.dim)
        individual += mutation
        individual = np.clip(individual, self.search_space[0], self.search_space[1])
        return individual

    def evaluate_fitness(self, func, individual):
        fitness = func(individual)
        return fitness

# Example usage:
def func(x):
    return np.sum(x**2)

hybrid_evolutionary_algorithm = HybridEvolutionaryAlgorithm(budget=100, dim=10)
best_solution = hybrid_evolutionary_algorithm(func)
print(f"Best solution: {best_solution}")