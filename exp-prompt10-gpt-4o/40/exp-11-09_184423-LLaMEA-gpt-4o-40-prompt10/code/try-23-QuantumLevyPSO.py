import numpy as np

class QuantumLevyPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.swarm_size = 40 + (dim * 4)  # Slightly reduced and scaled swarm size
        self.position = np.random.uniform(-5.0, 5.0, (self.swarm_size, dim))
        self.velocity = np.zeros((self.swarm_size, dim))
        self.personal_best_position = np.copy(self.position)
        self.personal_best_value = np.full(self.swarm_size, np.inf)
        self.global_best_position = np.zeros(dim)
        self.global_best_value = np.inf
        self.w_max = 0.8  # Adjusted inertia weight range
        self.w_min = 0.3
        self.c1_base = 1.5  # Tuned cognitive coefficients
        self.c2_base = 2.5
        self.chaos_factor = 0.2  # Reduced chaotic influence for stability
        self.alpha = 0.01  # Scaling factor for Lévy flight

    def levy_flight(self, dim):
        beta = 1.5
        sigma = (np.math.gamma(1 + beta) * np.sin(np.pi * beta / 2) /
                 (np.math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (1 / beta)
        u = np.random.normal(0, sigma, size=dim)
        v = np.random.normal(0, 1, size=dim)
        step = u / (np.abs(v) ** (1 / beta))
        return step

    def __call__(self, func):
        evals = 0
        while evals < self.budget:
            # Evaluate fitness
            fitness_values = np.array([func(pos) for pos in self.position])
            evals += self.swarm_size

            # Update personal bests
            better_mask = fitness_values < self.personal_best_value
            self.personal_best_value[better_mask] = fitness_values[better_mask]
            self.personal_best_position[better_mask] = self.position[better_mask]

            # Update global best
            min_fitness_idx = np.argmin(fitness_values)
            if fitness_values[min_fitness_idx] < self.global_best_value:
                self.global_best_value = fitness_values[min_fitness_idx]
                self.global_best_position = self.position[min_fitness_idx]

            # Update velocities and positions
            inertia_weight = self.w_max - ((self.w_max - self.w_min) * (evals / self.budget))
            r1 = np.random.rand(self.swarm_size, self.dim)
            r2 = np.random.rand(self.swarm_size, self.dim)
            
            # Dynamic coefficients
            c1 = self.c1_base * (1 - evals / self.budget)
            c2 = self.c2_base * (1 + evals / self.budget)

            cognitive_component = c1 * r1 * (self.personal_best_position - self.position)
            social_component = c2 * r2 * (self.global_best_position - self.position)
            self.velocity = (inertia_weight * self.velocity + cognitive_component + social_component)
            
            # Incorporate Lévy flight for exploration
            levy_step = np.array([self.levy_flight(self.dim) for _ in range(self.swarm_size)])
            new_position = self.position + self.velocity + self.alpha * levy_step

            # Enforce boundaries
            new_position = np.clip(new_position, -5.0, 5.0)
            self.position = new_position

        return self.global_best_position, self.global_best_value