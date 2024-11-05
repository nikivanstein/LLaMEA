import numpy as np

class HybridSwarmOptimizerACLOBLDIWSAMCM:
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
        self.mutation_rate = 0.1
        self.mutation_step_size = 0.1
        self.cauchy_mutation_rate = 0.05
        self.cauchy_mutation_scale = 1.0

    def levy_flight(self, size):
        r1 = np.random.uniform(size=size)
        r2 = np.random.uniform(size=size)
        return 0.01 * r1 / (r2 ** (1 / self.levy_flight_beta))

    def opposition_based_learning(self, position):
        return self.lower_bound + self.upper_bound - position

    def self_adaptive_mutation(self, position):
        mutation_mask = np.random.rand(self.dim) < self.mutation_rate
        mutation_vector = np.random.uniform(-self.mutation_step_size, self.mutation_step_size, size=self.dim)
        return position + mutation_vector * mutation_mask

    def cauchy_mutation(self, position):
        mutation_mask = np.random.rand(self.dim) < self.cauchy_mutation_rate
        mutation_vector = np.random.standard_cauchy(size=self.dim) * self.cauchy_mutation_scale
        return position + mutation_vector * mutation_mask

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
                # Self-adaptive mutation for increased diversity
                if np.random.rand() < self.mutation_rate:
                    mutated_position = self.self_adaptive_mutation(self.particles[i])
                    mutated_fitness = func(mutated_position)
                    evaluations += 1
                    if mutated_fitness < fitness:
                        self.particles[i] = mutated_position
                        self.best_fitness[i] = mutated_fitness
                        self.best_positions[i] = np.copy(self.particles[i])
                # Cauchy mutation for further exploration
                if np.random.rand() < self.cauchy_mutation_rate:
                    mutated_position = self.cauchy_mutation(self.particles[i])
                    mutated_fitness = func(mutated_position)
                    evaluations += 1
                    if mutated_fitness < fitness:
                        self.particles[i] = mutated_position
                        self.best_fitness[i] = mutated_fitness
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

optimizer = HybridSwarmOptimizerACLOBLDIWSAMCM(budget=1000, dim=10)
result = optimizer(example_func)
print(result)