import numpy as np

class EnhancedAdaptiveDE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 10 * dim
        self.bounds = (-5.0, 5.0)
        self.population = np.random.uniform(self.bounds[0], self.bounds[1], (self.pop_size, dim))
        self.fitness = np.full(self.pop_size, np.inf)
        self.CR = np.random.uniform(0.5, 1.0, self.pop_size)
        self.F = np.random.uniform(0.4, 0.9, self.pop_size)
        self.local_intensification = 0.45
        self.dynamic_scale = 0.35
        self.chaos_coefficient = 0.9
        self.learning_rate = 0.25
        self.memory = np.zeros(self.dim)
        self.topology_update_frequency = 10  # New: topology update frequency
        self.elite_fitness = np.inf
        self.elite_solution = None

    def chaotic_map(self, x):
        return self.chaos_coefficient * x * (1 - x)

    def update_topology(self):
        indices = np.argsort(self.fitness)
        elite_count = max(1, int(0.1 * self.pop_size))
        for i in range(elite_count, self.pop_size):
            self.population[indices[i]] = self.population[indices[np.random.randint(elite_count)]]

    def __call__(self, func):
        evaluations = 0
        chaos_value = np.random.rand()
        gen_count = 0
        while evaluations < self.budget:
            if gen_count % self.topology_update_frequency == 0:
                self.update_topology()
            for i in range(self.pop_size):
                if evaluations >= self.budget:
                    break
                indices = np.arange(self.pop_size)
                indices = indices[indices != i]
                a, b, c = np.random.choice(indices, 3, replace=False)
                dynamic_factor = 1.0 + self.dynamic_scale * (np.random.rand() - 0.5)
                mutant = self.population[a] + dynamic_factor * self.F[i] * (self.population[b] - self.population[c])

                if np.random.rand() < self.local_intensification:
                    local_best = self.population[np.argmin(self.fitness)]
                    mutant = (0.4 + chaos_value) * mutant + (0.6 - chaos_value) * (local_best + self.memory)
                
                mutant = np.clip(mutant, *self.bounds)

                j_rand = np.random.randint(self.dim)
                trial = np.where((np.random.rand(self.dim) < self.CR[i]) | (np.arange(self.dim) == j_rand), mutant, self.population[i])

                trial_fitness = func(trial)
                evaluations += 1
                if trial_fitness < self.fitness[i]:
                    self.memory = 0.7 * self.memory + 0.3 * (trial - self.population[i])
                    self.population[i] = trial
                    self.fitness[i] = trial_fitness
                    self.CR[i] = (1 - self.learning_rate) * self.CR[i] + self.learning_rate * np.random.rand()
                    self.F[i] = (1 - self.learning_rate) * self.F[i] + self.learning_rate * np.random.rand()
                    if trial_fitness < self.elite_fitness:
                        self.elite_fitness = trial_fitness
                        self.elite_solution = trial
                else:
                    self.CR[i] = (1 - self.learning_rate) * np.random.rand() + self.learning_rate * self.CR[i]
                    self.F[i] = (1 - self.learning_rate) * np.random.rand() + self.learning_rate * self.F[i]

                chaos_value = self.chaotic_map(chaos_value)
            gen_count += 1

        return self.elite_solution, self.elite_fitness