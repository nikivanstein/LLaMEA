import numpy as np

class AdaptiveParticleSwarmOptimization:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.bounds = (-5.0, 5.0)
        self.population_size = max(10, 10 * dim // 3)
        self.w = 0.5  # initial inertia weight
        self.c1 = 1.5  # cognitive component
        self.c2 = 1.5  # social component
        self.positions = np.random.uniform(self.bounds[0], self.bounds[1], (self.population_size, dim))
        self.velocities = np.random.uniform(-1, 1, (self.population_size, dim))
        self.personal_best_positions = np.copy(self.positions)
        self.personal_best_fitness = np.full(self.population_size, np.inf)
        self.global_best_position = None
        self.global_best_fitness = np.inf

    def __call__(self, func):
        evaluations = 0
        while evaluations < self.budget:
            fitness_values = np.apply_along_axis(func, 1, self.positions)
            evaluations += self.population_size
            
            for i in range(self.population_size):
                if fitness_values[i] < self.personal_best_fitness[i]:
                    self.personal_best_fitness[i] = fitness_values[i]
                    self.personal_best_positions[i] = self.positions[i]
                    
                if fitness_values[i] < self.global_best_fitness:
                    self.global_best_fitness = fitness_values[i]
                    self.global_best_position = self.positions[i]

            for i in range(self.population_size):
                r1, r2 = np.random.rand(2)
                self.velocities[i] = (self.w * self.velocities[i]
                                      + self.c1 * r1 * (self.personal_best_positions[i] - self.positions[i])
                                      + self.c2 * r2 * (self.global_best_position - self.positions[i]))
                # Dynamic velocity clamping
                self.velocities[i] = np.clip(self.velocities[i], -1, 1)
                self.positions[i] += self.velocities[i]
                self.positions[i] = np.clip(self.positions[i], *self.bounds)
            
            # Update inertia weight for exploration-exploitation balance
            self.w = max(0.4, self.w * 0.99)
        
        return self.global_best_position, self.global_best_fitness