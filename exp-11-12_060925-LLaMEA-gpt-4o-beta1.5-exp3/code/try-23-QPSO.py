import numpy as np

class QPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.evaluations = 0
        self.pop_size = 10 * dim
        self.beta = 0.5  # Contraction-expansion coefficient

    def __call__(self, func):
        # Initialize positions and personal bests
        positions = self.lower_bound + np.random.rand(self.pop_size, self.dim) * (self.upper_bound - self.lower_bound)
        pbest_positions = positions.copy()
        pbest_fitness = np.apply_along_axis(func, 1, positions)
        self.evaluations = self.pop_size

        # Find the global best
        gbest_idx = np.argmin(pbest_fitness)
        gbest_position = pbest_positions[gbest_idx]
        gbest_fitness = pbest_fitness[gbest_idx]

        while self.evaluations < self.budget:
            for i in range(self.pop_size):
                if self.evaluations >= self.budget:
                    break

                # Quantum-inspired update of positions
                p = np.random.rand(self.dim)
                mbest = (np.sum(pbest_positions, axis=0) / self.pop_size)
                u = np.random.rand(self.dim)
                new_position = pbest_positions[i] + self.beta * p * (mbest - np.abs(gbest_position - pbest_positions[i]) * np.log(1/u))

                # Apply boundary constraints
                new_position = np.clip(new_position, self.lower_bound, self.upper_bound)

                # Evaluate the new position
                new_fitness = func(new_position)
                self.evaluations += 1

                # Update personal best
                if new_fitness < pbest_fitness[i]:
                    pbest_positions[i] = new_position
                    pbest_fitness[i] = new_fitness

                    # Update global best
                    if new_fitness < gbest_fitness:
                        gbest_position = new_position
                        gbest_fitness = new_fitness

        return gbest_position, gbest_fitness