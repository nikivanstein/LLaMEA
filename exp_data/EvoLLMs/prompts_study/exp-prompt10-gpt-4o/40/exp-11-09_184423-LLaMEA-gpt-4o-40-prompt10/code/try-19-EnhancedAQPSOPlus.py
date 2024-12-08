import numpy as np

class EnhancedAQPSOPlus:
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
        self.w_min = 0.4
        self.c1_base = 2.0
        self.c2_base = 2.0
        self.learning_rate_decay = 0.98
        self.chaos_factor = 0.3

    def levy_flight(self):
        beta = 1.5
        sigma = (np.math.gamma(1 + beta) * np.sin(np.pi * beta / 2) /
                 (np.math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (1 / beta)
        u = np.random.normal(0, sigma, self.dim)
        v = np.random.normal(0, 1, self.dim)
        step = u / abs(v) ** (1 / beta)
        return step

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

            c1 = self.c1_base * (self.learning_rate_decay ** (evals / self.swarm_size))
            c2 = self.c2_base * (self.learning_rate_decay ** (evals / self.swarm_size))

            cognitive_component = c1 * r1 * (self.personal_best_position - self.position)
            social_component = c2 * r2 * (self.global_best_position - self.position)
            self.velocity = (inertia_weight * self.velocity + cognitive_component + social_component)

            chaos_component = self.chaos_factor * (np.random.rand(self.swarm_size, self.dim) - 0.5) * \
                              np.sin(1.5 * np.pi * evals / self.budget)

            levy_step = np.array([self.levy_flight() for _ in range(self.swarm_size)])

            new_position = self.position + self.velocity + chaos_component + levy_step

            new_position = np.clip(new_position, -5.0, 5.0)
            self.position = new_position

        return self.global_best_position, self.global_best_value