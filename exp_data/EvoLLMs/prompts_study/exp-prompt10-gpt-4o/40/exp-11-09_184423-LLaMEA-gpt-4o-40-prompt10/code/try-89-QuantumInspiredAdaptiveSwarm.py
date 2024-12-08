import numpy as np

class QuantumInspiredAdaptiveSwarm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.swarm_size = 40 + (dim * 4)  # Slightly smaller swarm for efficiency
        self.position = np.random.uniform(-5.0, 5.0, (self.swarm_size, dim))
        self.velocity = np.zeros((self.swarm_size, dim))
        self.personal_best_position = np.copy(self.position)
        self.personal_best_value = np.full(self.swarm_size, np.inf)
        self.global_best_position = np.zeros(dim)
        self.global_best_value = np.inf
        self.w_max = 0.8
        self.w_min = 0.3
        self.c1_base = 1.5
        self.c2_base = 2.5
        self.q_factor = 0.1  # Quantum potential factor
        self.adaptive_c1 = self.c1_base
        self.adaptive_c2 = self.c2_base

    def quantum_potential(self, size):
        q_field = np.random.normal(0, self.q_factor, size)
        return q_field

    def __call__(self, func):
        evals = 0
        while evals < self.budget:
            fitness_values = np.array([func(pos) for pos in self.position])
            evals += self.swarm_size

            better_mask = fitness_values < self.personal_best_value
            self.personal_best_value[better_mask] = fitness_values[better_mask]
            self.personal_best_position[better_mask] = self.position[better_mask]

            min_fitness_idx = np.argmin(fitness_values)
            if fitness_values[min_fitness_idx] < self.global_best_value:
                self.global_best_value = fitness_values[min_fitness_idx]
                self.global_best_position = self.position[min_fitness_idx]

            inertia_weight = self.w_max - ((self.w_max - self.w_min) * (evals / self.budget))
            r1 = np.random.rand(self.swarm_size, self.dim)
            r2 = np.random.rand(self.swarm_size, self.dim)

            self.adaptive_c1 = self.c1_base * (1.0 - evals / self.budget)
            self.adaptive_c2 = self.c2_base * (evals / self.budget)

            cognitive_component = self.adaptive_c1 * r1 * (self.personal_best_position - self.position)
            social_component = self.adaptive_c2 * r2 * (self.global_best_position - self.position)

            self.velocity = (inertia_weight * self.velocity + cognitive_component + social_component)

            quantum_component = self.quantum_potential((self.swarm_size, self.dim))
            new_position = self.position + self.velocity + quantum_component

            new_position = np.clip(new_position, -5.0, 5.0)
            self.position = new_position

        return self.global_best_position, self.global_best_value