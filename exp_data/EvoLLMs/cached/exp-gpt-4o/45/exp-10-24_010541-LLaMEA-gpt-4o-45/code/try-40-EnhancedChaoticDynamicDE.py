import numpy as np

class EnhancedChaoticDynamicDE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.initial_pop_size = 12 * dim
        self.bounds = (-5.0, 5.0)
        self.population = np.random.uniform(self.bounds[0], self.bounds[1], (self.initial_pop_size, dim))
        self.fitness = np.full(self.initial_pop_size, np.inf)
        self.CR = np.random.uniform(0.5, 1.0, self.initial_pop_size)
        self.F = np.random.uniform(0.4, 0.9, self.initial_pop_size)
        self.local_search_prob = 0.3  # updated local search probability
        self.dynamic_scale = 0.35  # enhanced dynamic scale factor
        self.chaos_coefficient = 0.85  # updated chaos coefficient
        self.learning_rate = 0.15  # increased learning rate

    def chaotic_map(self, x):
        return self.chaos_coefficient * x * (1 - x)

    def __call__(self, func):
        evaluations = 0
        chaos_value = np.random.rand()
        pop_size = self.initial_pop_size
        while evaluations < self.budget:
            for i in range(pop_size):
                if evaluations >= self.budget:
                    break
                if pop_size > self.budget - evaluations:
                    pop_size = self.budget - evaluations
                    self.population = self.population[:pop_size]
                    self.fitness = self.fitness[:pop_size]
                    
                # Mutation with dynamic scaling
                indices = np.arange(pop_size)
                indices = indices[indices != i]
                a, b, c = np.random.choice(indices, 3, replace=False)
                dynamic_factor = 1.0 + self.dynamic_scale * (np.random.rand() - 0.5)
                mutant = self.population[a] + dynamic_factor * self.F[i] * (self.population[b] - self.population[c])
                
                # Chaotic Local Search Intensification
                if np.random.rand() < self.local_search_prob:
                    local_best = self.population[np.argmin(self.fitness)]
                    mutant = (0.5 + chaos_value) * mutant + (0.5 - chaos_value) * local_best
                
                mutant = np.clip(mutant, *self.bounds)

                # Crossover
                j_rand = np.random.randint(self.dim)
                trial = np.where((np.random.rand(self.dim) < self.CR[i]) | (np.arange(self.dim) == j_rand), mutant, self.population[i])

                # Selection
                trial_fitness = func(trial)
                evaluations += 1
                if trial_fitness < self.fitness[i]:
                    self.population[i] = trial
                    self.fitness[i] = trial_fitness
                    self.CR[i] = (1 - self.learning_rate) * self.CR[i] + self.learning_rate * np.random.rand()  # adaptive adjustment
                    self.F[i] = (1 - self.learning_rate) * self.F[i] + self.learning_rate * np.random.rand()
                else:
                    self.CR[i] = (1 - self.learning_rate) * np.random.rand() + self.learning_rate * self.CR[i]
                    self.F[i] = (1 - self.learning_rate) * np.random.rand() + self.learning_rate * self.F[i]

                chaos_value = self.chaotic_map(chaos_value)

        best_idx = np.argmin(self.fitness)
        return self.population[best_idx], self.fitness[best_idx]