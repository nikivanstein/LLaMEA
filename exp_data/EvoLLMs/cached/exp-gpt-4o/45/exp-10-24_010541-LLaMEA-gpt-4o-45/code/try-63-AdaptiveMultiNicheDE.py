import numpy as np

class AdaptiveMultiNicheDE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 12 * dim
        self.bounds = (-5.0, 5.0)
        self.population = np.random.uniform(self.bounds[0], self.bounds[1], (self.pop_size, dim))
        self.fitness = np.full(self.pop_size, np.inf)
        self.CR = np.random.uniform(0.6, 1.0, self.pop_size)
        self.F = np.random.uniform(0.4, 0.8, self.pop_size)
        self.local_intensification = 0.4
        self.niche_radius = 0.1
        self.chaos_coefficient = 0.85
        self.learning_rate = 0.2
        self.memory = np.zeros(self.dim)

    def chaotic_map(self, x):
        return self.chaos_coefficient * x * (1 - x)

    def get_niche(self, ind):
        distances = np.linalg.norm(self.population - self.population[ind], axis=1)
        return distances < self.niche_radius

    def __call__(self, func):
        evaluations = 0
        chaos_value = np.random.rand()
        while evaluations < self.budget:
            for i in range(self.pop_size):
                if evaluations >= self.budget:
                    break
                indices = np.arange(self.pop_size)
                indices = indices[indices != i]
                a, b, c = np.random.choice(indices, 3, replace=False)
                niche_mask = self.get_niche(i)
                if np.any(niche_mask):
                    a, b, c = np.random.choice(np.where(niche_mask)[0], 3, replace=False)
                
                dynamic_factor = 1.0 + np.random.normal(0, 0.1)
                mutant = self.population[a] + dynamic_factor * self.F[i] * (self.population[b] - self.population[c])

                if np.random.rand() < self.local_intensification:
                    local_best = self.population[np.argmin(self.fitness)]
                    mutant = (0.5 + chaos_value) * mutant + (0.5 - chaos_value) * (local_best + self.memory)
                
                mutant = np.clip(mutant, *self.bounds)

                j_rand = np.random.randint(self.dim)
                trial = np.where((np.random.rand(self.dim) < self.CR[i]) | (np.arange(self.dim) == j_rand), mutant, self.population[i])

                trial_fitness = func(trial)
                evaluations += 1
                if trial_fitness < self.fitness[i]:
                    self.memory = 0.5 * self.memory + 0.5 * (trial - self.population[i])
                    self.population[i] = trial
                    self.fitness[i] = trial_fitness
                    self.CR[i] = (1 - self.learning_rate) * self.CR[i] + self.learning_rate * np.random.rand()
                    self.F[i] = (1 - self.learning_rate) * self.F[i] + self.learning_rate * np.random.rand()
                else:
                    self.CR[i] = (1 - self.learning_rate) * np.random.rand() + self.learning_rate * self.CR[i]
                    self.F[i] = (1 - self.learning_rate) * np.random.rand() + self.learning_rate * self.F[i]
                
                chaos_value = self.chaotic_map(chaos_value)

        best_idx = np.argmin(self.fitness)
        return self.population[best_idx], self.fitness[best_idx]