import numpy as np

class HybridEvolutionaryAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 100
        self.population = np.random.uniform(-5.0, 5.0, (self.population_size, self.dim))
        self.fitness_values = np.zeros(self.population_size)
        self.fittest_index = np.argmax(self.fitness_values)
        self.fittest_individual = self.population[self.fittest_index]

    def __call__(self, func):
        for _ in range(self.budget):
            # Evaluate fitness
            self.fitness_values = func(self.population)

            # Selection
            self.fittest_index = np.argmax(self.fitness_values)
            self.fittest_individual = self.population[self.fittest_index]

            # Crossover
            offspring = np.zeros((self.population_size, self.dim))
            for i in range(self.population_size):
                parent1 = self.population[i]
                parent2 = self.population[(i + 1) % self.population_size]
                child = (parent1 + parent2) / 2
                offspring[i] = child

            # Mutation
            for i in range(self.population_size):
                if np.random.rand() < 0.3:
                    mutation = np.random.uniform(-0.5, 0.5, self.dim)
                    offspring[i] += mutation

            # Adaptation
            self.population = offspring
            self.fitness_values = func(self.population)

            # Perturbation
            if np.random.rand() < 0.3:
                perturbation = np.random.uniform(-0.5, 0.5, self.dim)
                self.population += perturbation

# Example usage:
def func(x):
    return np.sum(x**2)

hybrid_ea = HybridEvolutionaryAlgorithm(budget=100, dim=10)
hybrid_ea( func )