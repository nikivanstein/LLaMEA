import numpy as np

class HybridSwarmOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.swarm_size = int(np.sqrt(budget))
        self.particles = np.random.uniform(self.lower_bound, self.upper_bound, size=(self.swarm_size, self.dim))
        self.velocities = np.zeros((self.swarm_size, self.dim))
        self.best_positions = np.copy(self.particles)
        self.best_fitness = np.inf * np.ones(self.swarm_size)
        self.global_best_position = np.random.uniform(self.lower_bound, self.upper_bound, size=self.dim)
        self.global_best_fitness = np.inf
        self.temperature = 1000.0
        self.cooling_rate = 0.99

    def __call__(self, func):
        evaluations = 0
        while evaluations < self.budget:
            for i in range(self.swarm_size):
                fitness = func(self.particles[i])
                evaluations += 1
                if fitness < self.best_fitness[i]:
                    self.best_fitness[i] = fitness
                    self.best_positions[i] = np.copy(self.particles[i])
                    if fitness < self.global_best_fitness:
                        self.global_best_fitness = fitness
                        self.global_best_position = np.copy(self.particles[i])
                self.velocities[i] += 0.5 * np.random.uniform(-1, 1, size=self.dim) + 0.5 * (self.best_positions[i] - self.particles[i]) + 0.5 * (self.global_best_position - self.particles[i])
                self.particles[i] += self.velocities[i]
                self.particles[i] = np.clip(self.particles[i], self.lower_bound, self.upper_bound)
                self.velocities[i] = np.clip(self.velocities[i], -1, 1)
            # Modified simulated annealing
            if evaluations % (self.swarm_size // 2) == 0:
                for i in range(self.swarm_size // 2):
                    new_position = np.copy(self.particles[i])
                    new_position += np.random.uniform(-1, 1, size=self.dim)
                    new_position = np.clip(new_position, self.lower_bound, self.upper_bound)
                    new_fitness = func(new_position)
                    evaluations += 1
                    if new_fitness < self.best_fitness[i] or np.random.rand() < np.exp((self.best_fitness[i] - new_fitness) / self.temperature):
                        self.particles[i] = new_position
                        self.best_fitness[i] = new_fitness
                        self.best_positions[i] = np.copy(self.particles[i])
                    self.temperature *= self.cooling_rate
        return self.global_best_position

# Example usage:
def example_func(x):
    return np.sum(x**2)

optimizer = HybridSwarmOptimizer(budget=1000, dim=10)
result = optimizer(example_func)
print(result)