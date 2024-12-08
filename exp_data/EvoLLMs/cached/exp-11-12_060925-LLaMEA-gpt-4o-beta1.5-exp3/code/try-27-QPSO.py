import numpy as np

class QPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.evaluations = 0
        self.swarm_size = 10 * dim
        self.alpha = 0.5  # control parameter for quantum behavior
        self.beta = 1.0   # control parameter for convergence speed

    def __call__(self, func):
        # Initialize the swarm
        swarm = self.lower_bound + np.random.rand(self.swarm_size, self.dim) * (self.upper_bound - self.lower_bound)
        fitness = np.apply_along_axis(func, 1, swarm)
        self.evaluations = self.swarm_size
        
        personal_best = swarm.copy()
        personal_best_fitness = fitness.copy()
        
        global_best_idx = np.argmin(fitness)
        global_best = swarm[global_best_idx]
        global_best_fitness = fitness[global_best_idx]
        
        while self.evaluations < self.budget:
            for i in range(self.swarm_size):
                if self.evaluations >= self.budget:
                    break

                # Calculate mean best position
                mean_best_position = np.mean(personal_best, axis=0)

                # Update position using quantum behavior
                phi = self.alpha * np.random.rand(self.dim)
                u = np.random.rand(self.dim)
                local_attraction = personal_best[i] - u * np.abs(global_best - mean_best_position)
                global_attraction = global_best - u * np.abs(global_best - mean_best_position)
                new_position = np.where(u < 0.5, local_attraction, global_attraction)

                # Control convergence speed
                new_position = new_position + self.beta * (np.random.rand(self.dim) - 0.5)

                # Ensure new positions are within bounds
                new_position = np.clip(new_position, self.lower_bound, self.upper_bound)

                # Evaluate new position
                new_fitness = func(new_position)
                self.evaluations += 1

                # Update personal and global bests
                if new_fitness < personal_best_fitness[i]:
                    personal_best[i] = new_position
                    personal_best_fitness[i] = new_fitness

                    if new_fitness < global_best_fitness:
                        global_best = new_position
                        global_best_fitness = new_fitness

        return global_best, global_best_fitness