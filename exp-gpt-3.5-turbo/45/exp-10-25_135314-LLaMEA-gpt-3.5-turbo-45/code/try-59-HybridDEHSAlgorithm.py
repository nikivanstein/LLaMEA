import numpy as np

class HybridDEHSAlgorithm:
    def __init__(self, budget, dim, harmony_memory_size=20, bandwidth=0.01, de_weight=0.8, de_cross_prob=0.9, de_pop_size=10, de_mut_prob=0.5):
        self.budget = budget
        self.dim = dim
        self.harmony_memory_size = harmony_memory_size
        self.bandwidth = bandwidth
        self.de_weight = de_weight
        self.de_cross_prob = de_cross_prob
        self.de_pop_size = de_pop_size
        self.de_mut_prob = de_mut_prob

    def initialize_population(self):
        return np.random.uniform(-5.0, 5.0, (self.harmony_memory_size, self.dim))

    def differential_evolution(self, population, func):
        mutant_pop = np.zeros_like(population)
        for i in range(self.harmony_memory_size):
            idxs = np.random.choice(self.harmony_memory_size, 3, replace=False)
            mutant = population[idxs[0]] + self.de_weight * (population[idxs[1]] - population[idxs[2])
            crossover = np.random.rand(self.dim) < self.de_cross_prob
            if not np.any(crossover):
                crossover[np.random.randint(0, self.dim)] = True
            mutant[crossover] = population[i][crossover]
            mutant_pop[i] = mutant
        return mutant_pop

    def harmony_search(self, population, func):
        new_population = np.copy(population)
        for i in range(self.harmony_memory_size):
            for j in range(self.dim):
                if np.random.rand() < self.bandwidth:
                    new_population[i][j] = np.random.uniform(-5.0, 5.0)
        return new_population

    def __call__(self, func):
        population = self.initialize_population()

        for _ in range(self.budget):
            de_population = self.differential_evolution(population, func)
            hs_population = self.harmony_search(population, func)
            combined_population = np.concatenate((population, de_population, hs_population))
            fitness_values = np.array([func(individual) for individual in combined_population])
            sorted_indices = np.argsort(fitness_values)
            population = combined_population[sorted_indices[:self.harmony_memory_size]]

        return population[np.argmin(np.array([func(individual) for individual in population]))]