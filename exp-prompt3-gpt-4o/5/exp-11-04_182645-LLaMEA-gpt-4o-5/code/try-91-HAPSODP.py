import numpy as np

class HAPSODP:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.bounds = (-5.0, 5.0)
        self.pop_size = 30
        self.w = 0.5  # inertia weight
        self.c1 = 1.5  # cognitive component
        self.c2 = 1.5  # social component
        self.mutation_factor = 0.9
        self.best_global_val = float('inf')
        self.best_global_pos = None
        self.population = np.random.uniform(self.bounds[0], self.bounds[1], (self.pop_size, self.dim))
        self.velocities = np.random.uniform(-1, 1, (self.pop_size, self.dim))
        self.personal_best_pos = np.copy(self.population)
        self.personal_best_val = np.full(self.pop_size, float('inf'))
        self.evaluations = 0
        self.restart_interval = self.budget // 10  # every 10% of the budget

    def __call__(self, func):
        while self.evaluations < self.budget:
            if self.evaluations % self.restart_interval == 0:
                # Reinitialize a portion of the population
                num_reinit = self.pop_size // 5
                indices = np.random.choice(self.pop_size, num_reinit, replace=False)
                self.population[indices] = np.random.uniform(self.bounds[0], self.bounds[1], (num_reinit, self.dim))

            for i in range(self.pop_size):
                if self.evaluations >= self.budget:
                    break

                # Evaluate fitness
                fitness = func(self.population[i])
                self.evaluations += 1

                # Update personal best
                if fitness < self.personal_best_val[i]:
                    self.personal_best_val[i] = fitness
                    self.personal_best_pos[i] = self.population[i]

                # Update global best
                if fitness < self.best_global_val:
                    self.best_global_val = fitness
                    self.best_global_pos = self.population[i]

            # Update velocities and positions
            for i in range(self.pop_size):
                r1, r2 = np.random.rand(2)
                self.velocities[i] = (self.w * self.velocities[i] +
                                     self.c1 * r1 * (self.personal_best_pos[i] - self.population[i]) +
                                     self.c2 * r2 * (self.best_global_pos - self.population[i]))

                # PSO position update
                self.population[i] = self.population[i] + self.velocities[i]
                self.population[i] = np.clip(self.population[i], self.bounds[0], self.bounds[1])

                # Adaptive mutation factor adjustment
                progress = self.evaluations / self.budget
                self.mutation_factor = 0.9 - 0.4 * progress

                # Apply differential perturbations
                if np.random.rand() < 0.5:
                    indices = np.random.choice(self.pop_size, 3, replace=False)
                    x0, x1, x2 = self.population[indices]
                    differential = self.mutation_factor * (x1 - x2)
                    candidate = np.clip(x0 + differential, self.bounds[0], self.bounds[1])
                    candidate_fitness = func(candidate)
                    self.evaluations += 1
                    if candidate_fitness < fitness:
                        self.population[i] = candidate

        return self.best_global_pos, self.best_global_val