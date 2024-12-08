import numpy as np

class DynamicMutPSOADE(HybridPSOADE):
    def __init__(self, budget, dim):
        super().__init__(budget, dim)
        self.base_mutation_rate = 0.1
        
    def __call__(self, func):
        for _ in range(self.budget):
            for i in range(self.population_size):
                fitness = func(self.particles[i])
                if fitness < self.best_fitness[i]:
                    self.best_fitness[i] = fitness
                    self.best_positions[i] = self.particles[i].copy()
            best_global_index = np.argmin(self.best_fitness)
            best_global_position = self.best_positions[best_global_index]
            for i in range(self.population_size):
                r1, r2 = np.random.uniform(0, 1, 2)
                self.velocities[i] = 0.5 * self.velocities[i] + 1.5 * r1 * (self.best_positions[i] - self.particles[i]) + 1.5 * r2 * (best_global_position - self.particles[i])
                new_position = self.particles[i] + self.velocities[i]
                new_position = np.clip(new_position, -5.0, 5.0)
                if func(new_position) < self.best_fitness[i]:
                    self.particles[i] = new_position
                    self.best_fitness[i] = func(new_position)
                if np.random.uniform() < self.base_mutation_rate * (1 - self.best_fitness[i] / np.max(self.best_fitness)):
                    mutation_direction = np.random.choice([-1, 1], size=self.dim)
                    mutation_step = np.random.uniform(0, 0.5)
                    mutated_position = self.particles[i] + mutation_step * mutation_direction
                    mutated_position = np.clip(mutated_position, -5.0, 5.0)
                    if func(mutated_position) < self.best_fitness[i]:
                        self.particles[i] = mutated_position
                        self.best_fitness[i] = func(mutated_position)