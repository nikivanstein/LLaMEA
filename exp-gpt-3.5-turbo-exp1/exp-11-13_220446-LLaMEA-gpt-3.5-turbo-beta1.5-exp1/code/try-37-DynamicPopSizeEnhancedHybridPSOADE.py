import numpy as np

class DynamicPopSizeEnhancedHybridPSOADE(EnhancedHybridPSOADE):
    def __init__(self, budget, dim):
        super().__init__(budget, dim)
        self.lower_bound_pop_size = 10
        self.upper_bound_pop_size = 50
        self.convergence_threshold = 0.5

    def __call__(self, func):
        for _ in range(self.budget):
            population_diversity = np.mean(np.std(self.particles, axis=0))
            self.mutation_rate = 0.1 + 0.1 * np.tanh(population_diversity - self.diversity_threshold)
            current_pop_size = self.lower_bound_pop_size + int((self.upper_bound_pop_size - self.lower_bound_pop_size) * np.tanh(population_diversity - self.convergence_threshold))
            self.update_population_size(current_pop_size)
            
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
                if np.random.uniform() < self.mutation_rate:
                    mutation_direction = np.random.choice([-1, 1], size=self.dim)
                    mutation_step = np.random.uniform(0, 0.5) + 0.2 * np.tanh(np.linalg.norm(self.velocities[i]))
                    mutated_position = self.particles[i] + mutation_step * mutation_direction
                    mutated_position = np.clip(mutated_position, -5.0, 5.0)
                    if func(mutated_position) < self.best_fitness[i]:
                        self.particles[i] = mutated_position
                        self.best_fitness[i] = func(mutated_position)

    def update_population_size(self, new_size):
        if new_size > self.population_size:
            self.particles = np.vstack((self.particles, np.random.uniform(-5.0, 5.0, (new_size - self.population_size, self.dim))))
            self.velocities = np.vstack((self.velocities, np.zeros((new_size - self.population_size, self.dim))))
            self.best_positions = np.vstack((self.best_positions, np.random.uniform(-5.0, 5.0, (new_size - self.population_size, self.dim))))
            self.best_fitness = np.hstack((self.best_fitness, np.full(new_size - self.population_size, np.inf)))
        elif new_size < self.population_size:
            indices_to_remove = np.random.choice(np.arange(self.population_size), size=self.population_size - new_size, replace=False)
            self.particles = np.delete(self.particles, indices_to_remove, axis=0)
            self.velocities = np.delete(self.velocities, indices_to_remove, axis=0)
            self.best_positions = np.delete(self.best_positions, indices_to_remove, axis=0)
            self.best_fitness = np.delete(self.best_fitness, indices_to_remove)
        self.population_size = new_size