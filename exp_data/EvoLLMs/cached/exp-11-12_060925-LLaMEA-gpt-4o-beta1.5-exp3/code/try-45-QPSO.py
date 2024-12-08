import numpy as np

class QPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.evaluations = 0
        self.population_size = 10 * dim
        self.alpha = 0.75  # coefficient for quantum-inspired update

    def __call__(self, func):
        position = self.lower_bound + np.random.rand(self.population_size, self.dim) * (self.upper_bound - self.lower_bound)
        velocity = np.zeros((self.population_size, self.dim))
        p_best = np.copy(position)
        p_best_fitness = np.apply_along_axis(func, 1, position)
        g_best_idx = np.argmin(p_best_fitness)
        g_best = position[g_best_idx]
        self.evaluations += self.population_size

        while self.evaluations < self.budget:
            for i in range(self.population_size):
                if self.evaluations >= self.budget:
                    break

                # Quantum-inspired position update
                beta = np.random.rand(self.dim)
                mbest = np.mean(p_best, axis=0)
                u = np.random.uniform(self.lower_bound, self.upper_bound, self.dim)
                position[i] += self.alpha * (beta * (g_best - np.abs(position[i] - mbest)) + (1 - beta) * (u - position[i]))
                position[i] = np.clip(position[i], self.lower_bound, self.upper_bound)

                fitness = func(position[i])
                self.evaluations += 1

                # Update personal best and global best
                if fitness < p_best_fitness[i]:
                    p_best_fitness[i] = fitness
                    p_best[i] = position[i]

                if fitness < p_best_fitness[g_best_idx]:
                    g_best_idx = i
                    g_best = position[i]

        return g_best, p_best_fitness[g_best_idx]