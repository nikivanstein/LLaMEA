import numpy as np

class QPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.evaluations = 0
        self.population_size = 10 * dim
        self.alpha = 0.75 # control parameter for convergence speed
        self.beta = 0.1 # control parameter for impact of global best

    def __call__(self, func):
        # Initialize the particle positions
        population = self.lower_bound + np.random.rand(self.population_size, self.dim) * (self.upper_bound - self.lower_bound)
        fitness = np.apply_along_axis(func, 1, population)
        self.evaluations = self.population_size

        # Initialize personal and global bests
        personal_best_positions = population.copy()
        personal_best_fitness = fitness.copy()
        global_best_idx = np.argmin(fitness)
        global_best_position = population[global_best_idx]
        global_best_fitness = fitness[global_best_idx]

        while self.evaluations < self.budget:
            for i in range(self.population_size):
                if self.evaluations >= self.budget:
                    break

                # Quantum-inspired position update
                p = np.random.rand(self.dim)
                u = np.random.rand(self.dim)
                mbest = np.mean(personal_best_positions, axis=0)
                theta = 2 * np.pi * p
                r = np.abs(global_best_position - personal_best_positions[i])
                step = self.alpha * u * r * np.cos(theta) + self.beta * (global_best_position - self.dim / 2)
                new_position = personal_best_positions[i] + step
                new_position = np.clip(new_position, self.lower_bound, self.upper_bound)

                # Evaluate the new position
                new_fitness = func(new_position)
                self.evaluations += 1

                # Update personal and global bests
                if new_fitness < personal_best_fitness[i]:
                    personal_best_positions[i] = new_position
                    personal_best_fitness[i] = new_fitness
                    if new_fitness < global_best_fitness:
                        global_best_position = new_position
                        global_best_fitness = new_fitness

        return global_best_position, global_best_fitness