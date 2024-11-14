import numpy as np

class DynamicHistoricalQuantumSwarm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.swarm_size = 50 + (dim * 5)
        self.position = np.random.uniform(-5.0, 5.0, (self.swarm_size, dim))
        self.velocity = np.zeros((self.swarm_size, dim))
        self.personal_best_position = np.copy(self.position)
        self.personal_best_value = np.full(self.swarm_size, np.inf)
        self.global_best_position = np.zeros(dim)
        self.global_best_value = np.inf
        self.phi = 0.5
        self.w_max = 0.9
        self.w_min = 0.3  # Adjusted for quicker convergence
        self.c1_base = 2.0
        self.c2_base = 1.5  # Adjusted for balance
        self.chaos_factor = 0.25
        self.levy_beta = 1.5
        self.dynamic_swarm_factor = 0.2  # Enhanced for dynamic exploration
        self.memory_factor = 0.2  # Enhanced historical memory
        self.quantum_factor = 0.1  # New: Quantum position update

    def levy_flight(self, size):
        sigma1 = pow((1.0 + self.levy_beta) * np.math.gamma((1.0 + self.levy_beta) / 2.0) /
                     (np.math.gamma(1.0 + self.levy_beta) * self.levy_beta * pow(2.0, ((self.levy_beta - 1.0) / 2.0))), 1.0 / self.levy_beta)
        u = np.random.normal(0, sigma1, size)
        v = np.random.normal(0, 1, size)
        step = u / (np.abs(v) ** (1.0 / self.levy_beta))
        return step

    def __call__(self, func):
        evals = 0
        historical_best = np.copy(self.global_best_position)
        historical_best_value = np.inf
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

            if self.global_best_value < historical_best_value:
                historical_best_value = self.global_best_value
                historical_best = np.copy(self.global_best_position)

            inertia_weight = self.w_max - ((self.w_max - self.w_min) * (evals / self.budget))
            r1 = np.random.rand(self.swarm_size, self.dim)
            r2 = np.random.rand(self.swarm_size, self.dim)
            r3 = np.random.rand(self.swarm_size, self.dim)

            adaptive_c1 = self.c1_base * (1.0 - evals / self.budget)
            adaptive_c2 = self.c2_base * (evals / self.budget)

            cognitive_component = adaptive_c1 * r1 * (self.personal_best_position - self.position)
            social_component = adaptive_c2 * r2 * (self.global_best_position - self.position)
            memory_component = self.memory_factor * r3 * (historical_best - self.position)

            self.velocity = (inertia_weight * self.velocity + cognitive_component + social_component + memory_component)

            chaos_component = self.chaos_factor * (np.random.rand(self.swarm_size, self.dim) - 0.5) * \
                              np.sin(1.5 * np.pi * evals / self.budget)
            new_position = self.position + self.velocity + chaos_component

            levy_component = 0.01 * self.levy_flight((self.swarm_size, self.dim))
            new_position += levy_component

            # Quantum-inspired update
            quantum_jump = self.quantum_factor * np.sign(np.random.rand(self.swarm_size, self.dim) - 0.5) * \
                           np.abs(self.position - self.global_best_position)
            new_position += quantum_jump

            disturbance = (np.random.rand(self.swarm_size, self.dim) - 0.5) * \
                          self.dynamic_swarm_factor * np.sin(evals / (self.budget / 4.0))
            new_position += disturbance

            new_position = np.clip(new_position, -5.0, 5.0)
            self.position = new_position

        return self.global_best_position, self.global_best_value