import numpy as np

class HybridDEwithDynamicScaling:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = 10 * dim
        self.population = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, dim))
        self.fitness = np.full(self.population_size, np.inf)
        self.scale_factor = 0.5
        self.crossover_rate = 0.9
        self.evaluations = 0

    def __call__(self, func):
        self.evaluate_population(func)
        while self.evaluations < self.budget:
            for i in range(self.population_size):
                if self.evaluations >= self.budget:
                    break
                trial_vector = self.mutate(i)
                trial_vector = self.crossover(trial_vector, self.population[i])
                trial_vector = self.adaptive_local_search(trial_vector, func)
                trial_fitness = func(trial_vector)
                self.evaluations += 1
                if trial_fitness < self.fitness[i]:
                    self.fitness[i] = trial_fitness
                    self.population[i] = trial_vector
            self.dynamic_scaling()

        best_idx = np.argmin(self.fitness)
        return self.population[best_idx]

    def evaluate_population(self, func):
        for i in range(self.population_size):
            if self.evaluations < self.budget:
                self.fitness[i] = func(self.population[i])
                self.evaluations += 1

    def mutate(self, idx):
        indices = [i for i in range(self.population_size) if i != idx]
        a, b, c = np.random.choice(indices, 3, replace=False)
        mutant_vector = self.population[a] + self.scale_factor * (self.population[b] - self.population[c])
        return np.clip(mutant_vector, self.lower_bound, self.upper_bound)

    def crossover(self, mutant_vector, target_vector):
        crossover = np.random.rand(self.dim) < self.crossover_rate
        if not np.any(crossover):
            crossover[np.random.randint(0, self.dim)] = True
        trial_vector = np.where(crossover, mutant_vector, target_vector)
        return trial_vector

    def adaptive_local_search(self, vector, func):
        best_vector = vector
        best_fitness = func(vector)
        for _ in range(3):
            local_vector = vector + np.random.normal(0, 0.1, self.dim)
            local_vector = np.clip(local_vector, self.lower_bound, self.upper_bound)
            local_fitness = func(local_vector)
            if local_fitness < best_fitness:
                best_fitness = local_fitness
                best_vector = local_vector
        return best_vector

    def dynamic_scaling(self):
        self.scale_factor = 0.5 + 0.3 * np.random.rand()