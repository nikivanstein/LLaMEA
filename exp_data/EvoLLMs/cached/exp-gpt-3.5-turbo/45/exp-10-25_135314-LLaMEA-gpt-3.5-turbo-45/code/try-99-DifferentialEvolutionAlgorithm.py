import numpy as np

class DifferentialEvolutionAlgorithm:
    def __init__(self, budget, dim, population_size=50, F=0.5, CR=0.9):
        self.budget = budget
        self.dim = dim
        self.population_size = population_size
        self.F = F
        self.CR = CR

    def initialize_population(self):
        return np.random.uniform(-5.0, 5.0, (self.population_size, self.dim))

    def mutate(self, population, target_idx):
        candidates = [idx for idx in range(self.population_size) if idx != target_idx]
        np.random.shuffle(candidates)
        a, b, c = candidates[:3]
        return population[a] + self.F * (population[b] - population[c])

    def crossover(self, target_vector, trial_vector):
        crossover_points = np.random.rand(self.dim) < self.CR
        new_vector = np.where(crossover_points, trial_vector, target_vector)
        return new_vector

    def select_population(self, target_vector, trial_vector, func):
        target_fitness = func(target_vector)
        trial_fitness = func(trial_vector)
        return trial_vector if trial_fitness < target_fitness else target_vector

    def __call__(self, func):
        population = self.initialize_population()

        for _ in range(self.budget):
            new_population = np.empty_like(population)
            for idx, target_vector in enumerate(population):
                mutated_vector = self.mutate(population, idx)
                trial_vector = self.crossover(target_vector, mutated_vector)
                new_population[idx] = self.select_population(target_vector, trial_vector, func)
            population = new_population

        return population[np.argmin([func(individual) for individual in population])]