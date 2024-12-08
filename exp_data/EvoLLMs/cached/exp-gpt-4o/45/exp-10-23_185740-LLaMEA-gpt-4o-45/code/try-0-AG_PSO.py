import numpy as np

class AG_PSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.bounds = (-5.0, 5.0)
        self.population_size = 100
        self.c1 = 1.5  # cognitive coefficient
        self.c2 = 1.5  # social coefficient
        self.inertia_weight = 0.7
        self.epsilon = 1e-8  # small value to prevent division by zero

    def __call__(self, func):
        np.random.seed(42)
        lower_bound, upper_bound = self.bounds
        population = np.random.uniform(lower_bound, upper_bound, (self.population_size, self.dim))
        velocities = np.random.uniform(-1, 1, (self.population_size, self.dim))
        personal_best = np.copy(population)
        personal_best_values = np.array([func(ind) for ind in population])
        global_best_idx = np.argmin(personal_best_values)
        global_best = personal_best[global_best_idx]
        
        evaluations = self.population_size

        while evaluations < self.budget:
            # Compute gradients for all particles
            gradient = np.zeros_like(population)
            for i in range(self.population_size):
                original_value = func(population[i])
                for d in range(self.dim):
                    step = np.zeros(self.dim)
                    step[d] = self.epsilon
                    gradient[i][d] = (func(population[i] + step) - original_value) / self.epsilon
                    evaluations += 1
                    if evaluations >= self.budget:
                        break
                if evaluations >= self.budget:
                    break

            # Update velocities and positions
            r1, r2 = np.random.rand(), np.random.rand()
            velocities = (self.inertia_weight * velocities +
                          self.c1 * r1 * (personal_best - population) +
                          self.c2 * r2 * (global_best - population) -
                          0.01 * gradient)  # Incorporating gradient
            population += velocities

            # Enforce bounds
            population = np.clip(population, lower_bound, upper_bound)

            # Evaluate and update personal and global bests
            for i in range(self.population_size):
                if evaluations < self.budget:
                    fitness = func(population[i])
                    evaluations += 1
                    if fitness < personal_best_values[i]:
                        personal_best_values[i] = fitness
                        personal_best[i] = population[i]
                    if fitness < personal_best_values[global_best_idx]:
                        global_best_idx = i
                        global_best = personal_best[i]
                else:
                    break

        return global_best