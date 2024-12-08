import numpy as np

class HybridPSOandDE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = 10 * dim
        self.population = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, dim))
        self.fitness = np.full(self.population_size, np.inf)
        self.scale_factor = 0.8
        self.crossover_rate = 0.9
        self.evaluations = 0
        self.velocities = np.zeros((self.population_size, dim))
        self.personal_best = np.copy(self.population)
        self.best_fitness = np.full(self.population_size, np.inf)
        self.global_best = None

    def __call__(self, func):
        self.evaluate_population(func)
        while self.evaluations < self.budget:
            for i in range(self.population_size):
                if self.evaluations >= self.budget:
                    break
                self.update_velocity(i)
                self.update_position(i)
                trial_vector = self.mutate(i)
                trial_vector = self.crossover(trial_vector, self.population[i])
                trial_vector = self.local_search(trial_vector)
                trial_fitness = func(trial_vector)
                self.evaluations += 1
                if trial_fitness < self.fitness[i]:
                    self.fitness[i] = trial_fitness
                    self.population[i] = trial_vector
                    self.personal_best[i] = trial_vector
                    self.best_fitness[i] = trial_fitness
                if self.global_best is None or trial_fitness < func(self.global_best):
                    self.global_best = trial_vector
        best_idx = np.argmin(self.fitness)
        return self.population[best_idx]

    def evaluate_population(self, func):
        for i in range(self.population_size):
            if self.evaluations < self.budget:
                self.fitness[i] = func(self.population[i])
                self.best_fitness[i] = self.fitness[i]
                self.personal_best[i] = self.population[i]
                self.evaluations += 1

    def update_velocity(self, idx):
        inertia_weight = 0.5
        cognitive_coeff = 2.0
        social_coeff = 2.0
        self.velocities[idx] = (inertia_weight * self.velocities[idx] +
                                cognitive_coeff * np.random.rand(self.dim) * (self.personal_best[idx] - self.population[idx]) +
                                social_coeff * np.random.rand(self.dim) * (self.global_best - self.population[idx]))

    def update_position(self, idx):
        self.population[idx] = np.clip(self.population[idx] + self.velocities[idx], self.lower_bound, self.upper_bound)

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

    def local_search(self, vector):
        local_vector = vector + np.random.normal(0, 0.1, self.dim)
        return np.clip(local_vector, self.lower_bound, self.upper_bound)