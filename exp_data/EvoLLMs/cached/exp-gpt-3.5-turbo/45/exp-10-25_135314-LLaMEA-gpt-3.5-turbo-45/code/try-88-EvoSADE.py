import numpy as np

class EvoSADE:
    def __init__(self, budget, dim, pop_size=50, f=0.5, cr=0.9):
        self.budget = budget
        self.dim = dim
        self.pop_size = pop_size
        self.f = f
        self.cr = cr

    def initialize_population(self):
        return np.random.uniform(-5.0, 5.0, (self.pop_size, self.dim))

    def mutate(self, population, target_idx, scaling_factor):
        candidates = np.array([idx for idx in range(self.pop_size) if idx != target_idx])
        selected = np.random.choice(candidates, 3, replace=False)
        mutant_vector = population[selected[0]] + scaling_factor * (population[selected[1]] - population[selected[2]])
        return mutant_vector

    def crossover(self, target_vector, mutant_vector):
        crossover_points = np.random.rand(self.dim) < self.cr
        trial_vector = np.where(crossover_points, mutant_vector, target_vector)
        return trial_vector

    def __call__(self, func):
        population = self.initialize_population()
        fitness = np.array([func(individual) for individual in population])
        best_idx = np.argmin(fitness)

        for _ in range(self.budget):
            target_idx = np.random.randint(self.pop_size)
            scaling_factor = np.random.uniform(0.5, 1.0)
            mutant_vector = self.mutate(population, target_idx, scaling_factor)
            trial_vector = self.crossover(population[target_idx], mutant_vector)
            trial_fitness = func(trial_vector)

            if trial_fitness < fitness[target_idx]:
                population[target_idx] = trial_vector
                fitness[target_idx] = trial_fitness
                if trial_fitness < fitness[best_idx]:
                    best_idx = target_idx

        return population[best_idx]