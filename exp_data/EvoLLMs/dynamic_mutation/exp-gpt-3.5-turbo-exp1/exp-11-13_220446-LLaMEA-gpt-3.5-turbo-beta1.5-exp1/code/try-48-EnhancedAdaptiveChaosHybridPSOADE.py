import numpy as np

class EnhancedAdaptiveChaosHybridPSOADE(AdaptiveHybridPSOADE):
    def __init__(self, budget, dim):
        super().__init__(budget, dim)
        
    def __call__(self, func):
        for _ in range(self.budget):
            population_diversity = np.mean(np.std(self.particles, axis=0))
            self.mutation_rate = 0.1 + 0.1 * np.tanh(population_diversity - self.diversity_threshold)
            inertia_weight = 0.5 + 0.3 * np.tanh(np.mean(self.best_fitness) - func(self.best_positions[np.argmin(self.best_fitness)]))
            for i in range(self.population_size):
                fitness = func(self.particles[i])
                if fitness < self.best_fitness[i]:
                    self.best_fitness[i] = fitness
                    self.best_positions[i] = self.particles[i].copy()
            best_global_index = np.argmin(self.best_fitness)
            best_global_position = self.best_positions[best_global_index]
            for i in range(self.population_size):
                r1, r2 = np.random.uniform(0, 1, 2)
                chaos_factor = np.random.normal(0.5, 0.1, size=self.dim)
                self.velocities[i] = inertia_weight * self.velocities[i] + 1.5 * r1 * (self.best_positions[i] - self.particles[i]) + 1.5 * r2 * (best_global_position - self.particles[i]) + chaos_factor
                new_position = self.particles[i] + self.velocities[i]
                new_position = np.clip(new_position, -5.0, 5.0)
                if func(new_position) < self.best_fitness[i]:
                    self.particles[i] = new_position
                    self.best_fitness[i] = func(new_position)
                if np.random.uniform() < self.mutation_rate:
                    mutation_direction = np.random.choice([-1, 1], size=self.dim)
                    mutation_step = np.random.uniform(0, 0.5) + 0.2 * np.tanh(np.linalg.norm(self.velocities[i])) + 0.1 * np.tanh(func(self.particles[i]) - np.mean(self.best_fitness))
                    mutated_position = self.particles[i] + mutation_step * mutation_direction
                    mutated_position = np.clip(mutated_position, -5.0, 5.0)
                    if func(mutated_position) < self.best_fitness[i]:
                        self.particles[i] = mutated_position
                        self.best_fitness[i] = func(mutated_position)