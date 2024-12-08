import numpy as np

class EnhancedChaoticDE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 10 * dim  # Increased population size for broader search
        self.bounds = (-5.0, 5.0)
        self.population = np.random.uniform(self.bounds[0], self.bounds[1], (self.pop_size, dim))
        self.fitness = np.full(self.pop_size, np.inf)
        self.CR = np.random.uniform(0.3, 0.9, self.pop_size)  # Narrowed CR range for controlled variation
        self.F = np.random.uniform(0.2, 0.7, self.pop_size)  # Narrowed F range for precision
        self.local_intensification = 0.35  # Adjusted local search probability
        self.dynamic_scale = 0.25  # Further reduced dynamic scale for stability
        self.chaos_coefficient = 0.9  # Increased chaos for greater exploratory diversity
        self.learning_rate = 0.35  # Increased adaptation speed for rapid convergence
        self.memory = np.zeros(self.dim)  # Memory vector for temporal learning

    def chaotic_map(self, x):
        return self.chaos_coefficient * x * (1 - x)

    def stochastic_perturbation(self, vector):
        perturbation_strength = np.random.uniform(-0.05, 0.05, self.dim)
        return vector + perturbation_strength

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
                dynamic_factor = 1.0 + self.dynamic_scale * (np.random.rand() - 0.5)
                mutant = self.population[a] + dynamic_factor * self.F[i] * (self.population[b] - self.population[c])

                if np.random.rand() < self.local_intensification:
                    local_best = self.population[np.argmin(self.fitness)]
                    mutant = (0.5 + chaos_value) * mutant + (0.5 - chaos_value) * (local_best + self.memory)

                mutant = self.stochastic_perturbation(mutant)
                mutant = np.clip(mutant, *self.bounds)

                j_rand = np.random.randint(self.dim)
                trial = np.where((np.random.rand(self.dim) < self.CR[i]) | (np.arange(self.dim) == j_rand), mutant, self.population[i])

                trial_fitness = func(trial)
                evaluations += 1
                if trial_fitness < self.fitness[i]:
                    self.memory = 0.5 * self.memory + 0.5 * (trial - self.population[i])  # Optimized memory weight
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