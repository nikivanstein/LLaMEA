import numpy as np

class HybridSwarmOptimizerACLOBLDIWSADE:
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
        self.adaptive_cooling_rate = 0.5
        self.levy_flight_alpha = 1.5
        self.levy_flight_beta = 1.8
        self.opposition_based_learning_rate = 0.2
        self.inertia_weight = 0.9
        self.inertia_weight_damping_ratio = 0.99
        self.de_scale_factor = 0.5
        self.de_crossover_probability = 0.5

    def levy_flight(self, size):
        r1 = np.random.uniform(size=size)
        r2 = np.random.uniform(size=size)
        return 0.01 * r1 / (r2 ** (1 / self.levy_flight_beta))

    def opposition_based_learning(self, position):
        return self.lower_bound + self.upper_bound - position

    def differential_evolution(self, position):
        r1, r2 = np.random.choice(self.swarm_size, 2, replace=False)
        mutant = self.particles[r1] + self.de_scale_factor * (self.particles[r2] - position)
        mutant = np.clip(mutant, self.lower_bound, self.upper_bound)
        crossover = np.where(np.random.rand(self.dim) < self.de_crossover_probability, mutant, position)
        return crossover

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
                self.velocities[i] = self.inertia_weight * self.velocities[i] + 0.5 * np.random.uniform(-1, 1, size=self.dim) + 0.5 * (self.best_positions[i] - self.particles[i]) + 0.5 * (self.global_best_position - self.particles[i])
                self.particles[i] += self.velocities[i]
                self.particles[i] = np.clip(self.particles[i], self.lower_bound, self.upper_bound)
                self.velocities[i] = np.clip(self.velocities[i], -1, 1)
                # Levy flight for enhanced global search
                if np.random.rand() < 0.2:
                    self.particles[i] += self.levy_flight(self.dim)
                    self.particles[i] = np.clip(self.particles[i], self.lower_bound, self.upper_bound)
                # Opposition-based learning for improved convergence
                if np.random.rand() < self.opposition_based_learning_rate:
                    opposition_position = self.opposition_based_learning(self.particles[i])
                    opposition_fitness = func(opposition_position)
                    evaluations += 1
                    if opposition_fitness < fitness:
                        self.particles[i] = opposition_position
                        self.best_fitness[i] = opposition_fitness
                        self.best_positions[i] = np.copy(self.particles[i])
                # Self-adaptive differential evolution for enhanced local search
                if np.random.rand() < 0.1:
                    de_position = self.differential_evolution(self.particles[i])
                    de_fitness = func(de_position)
                    evaluations += 1
                    if de_fitness < fitness:
                        self.particles[i] = de_position
                        self.best_fitness[i] = de_fitness
                        self.best_positions[i] = np.copy(self.particles[i])
            # Modified simulated annealing with adaptive cooling
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
                    self.temperature *= self.cooling_rate * (1 - self.adaptive_cooling_rate * (evaluations / self.budget))
            # Dynamic inertia weight
            self.inertia_weight *= self.inertia_weight_damping_ratio
        return self.global_best_position

# Example usage:
def example_func(x):
    return np.sum(x**2)

optimizer = HybridSwarmOptimizerACLOBLDIWSADE(budget=1000, dim=10)
result = optimizer(example_func)
print(result)