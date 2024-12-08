import numpy as np

class AdaptiveDifferentialEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.bounds = (-5.0, 5.0)
        self.pop_size = 10 * dim
        self.F = 0.8  # Differential weight
        self.CR = 0.9  # Crossover probability
        self.evaluations = 0
        self.learning_rate = 0.05  # Learning rate for adaptive parameters

    def initialize_population(self):
        return np.random.uniform(self.bounds[0], self.bounds[1], (self.pop_size, self.dim))

    def mutate(self, idx, population):
        candidates = list(range(self.pop_size))
        candidates.remove(idx)
        a, b, c = np.random.choice(candidates, 3, replace=False)
        # Introduce dynamic scaling factor for enhanced exploration
        rand_scale = np.random.uniform(0.7, 1.3)
        mutant = np.clip(population[a] + self.F * rand_scale * (population[b] - population[c]), self.bounds[0], self.bounds[1])
        return mutant

    def crossover(self, target, mutant):
        crossover_mask = np.random.rand(self.dim) < self.CR
        return np.where(crossover_mask, mutant, target)

    def select(self, candidate, target, func):
        if func(candidate) < func(target):
            return candidate
        return target

    def adapt_parameters(self):
        self.F += self.learning_rate * (np.random.uniform(-0.1, 0.1))
        self.F = np.clip(self.F, 0.5, 1.0)
        # Dynamic CR adaptation for robust convergence
        self.CR += self.learning_rate * (np.random.uniform(-0.05, 0.05))
        self.CR = np.clip(self.CR, 0.7, 1.0)  # Adjusted for more rigorous exploration

    def __call__(self, func):
        population = self.initialize_population()
        fitness = np.apply_along_axis(func, 1, population)
        self.evaluations += self.pop_size

        while self.evaluations < self.budget:
            for i in range(self.pop_size):
                self.adapt_parameters()  # Adaptive parameters
                mutant = self.mutate(i, population)
                trial = self.crossover(population[i], mutant)
                if self.evaluations < self.budget:
                    trial_fitness = func(trial)
                    self.evaluations += 1
                    if trial_fitness < fitness[i]:
                        fitness[i] = trial_fitness
                        population[i] = trial

        best_idx = np.argmin(fitness)
        return population[best_idx]