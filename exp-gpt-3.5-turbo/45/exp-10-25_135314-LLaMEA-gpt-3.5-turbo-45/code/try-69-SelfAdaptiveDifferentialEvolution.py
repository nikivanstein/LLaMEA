import numpy as np

class SelfAdaptiveDifferentialEvolution:
    def __init__(self, budget, dim, population_size=30, scaling_factor=0.5, crossover_prob=0.9):
        self.budget = budget
        self.dim = dim
        self.population_size = population_size
        self.scaling_factor = scaling_factor
        self.crossover_prob = crossover_prob
        self.population = np.random.uniform(-5.0, 5.0, (population_size, dim))

    def mutate(self, target_idx, iter_count):
        scaling_factor = self.scaling_factor * np.exp(-4 * iter_count / self.budget)
        candidates = [idx for idx in range(self.population_size) if idx != target_idx]
        r1, r2, r3 = np.random.choice(candidates, 3, replace=False)
        mutant_vector = self.population[r1] + scaling_factor * (self.population[r2] - self.population[r3])
        return mutant_vector

    def crossover(self, target_vector, mutant_vector):
        trial_vector = np.copy(target_vector)
        crossover_points = np.random.rand(self.dim) < self.crossover_prob
        trial_vector[crossover_points] = mutant_vector[crossover_points]
        return trial_vector

    def __call__(self, func):
        for iter_count in range(self.budget):
            for idx, target_vector in enumerate(self.population):
                mutant_vector = self.mutate(idx, iter_count)
                trial_vector = self.crossover(target_vector, mutant_vector)
                target_fitness = func(target_vector)
                trial_fitness = func(trial_vector)

                if trial_fitness < target_fitness:
                    self.population[idx] = trial_vector

        best_idx = np.argmin([func(individual) for individual in self.population])
        return self.population[best_idx]