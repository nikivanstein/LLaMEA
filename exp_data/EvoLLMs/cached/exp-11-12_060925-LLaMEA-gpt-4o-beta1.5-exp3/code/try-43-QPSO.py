import numpy as np

class QPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.evaluations = 0
        self.swarm_size = 10 * dim
        self.phi = 0.5  # Contraction-expansion coefficient

    def __call__(self, func):
        # Initialize the swarm
        position = self.lower_bound + np.random.rand(self.swarm_size, self.dim) * (self.upper_bound - self.lower_bound)
        personal_best_position = position.copy()
        fitness = np.apply_along_axis(func, 1, position)
        personal_best_fitness = fitness.copy()

        self.evaluations = self.swarm_size

        # Identify the global best
        global_best_idx = np.argmin(personal_best_fitness)
        global_best_position = personal_best_position[global_best_idx]
        global_best_fitness = personal_best_fitness[global_best_idx]

        while self.evaluations < self.budget:
            for i in range(self.swarm_size):
                if self.evaluations >= self.budget:
                    break

                # Calculate the mean best position
                mbest = np.mean(personal_best_position, axis=0)

                # Generate a random number array
                u = np.random.rand(self.dim)

                # Update particle's position using quantum-behavior inspired formula
                p = (1 - self.phi) * personal_best_position[i] + self.phi * global_best_position
                b = np.sign(u - 0.5) * np.log(1.0 / (1.0 - u))
                new_position = p + b * np.abs(mbest - personal_best_position[i])

                # Ensure the new position is within bounds
                new_position = np.clip(new_position, self.lower_bound, self.upper_bound)

                # Evaluate the new fitness
                fitness_value = func(new_position)
                self.evaluations += 1

                # Update personal best if the new fitness is better
                if fitness_value < personal_best_fitness[i]:
                    personal_best_position[i] = new_position
                    personal_best_fitness[i] = fitness_value

                    # Update global best if the new personal best is better
                    if fitness_value < global_best_fitness:
                        global_best_position = new_position
                        global_best_fitness = fitness_value

        return global_best_position, global_best_fitness