import numpy as np

class EnhancedAQPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.initial_swarm_size = 50 + (dim * 5)
        self.position = np.random.uniform(-5.0, 5.0, (self.initial_swarm_size, dim))
        self.velocity = np.zeros((self.initial_swarm_size, dim))
        self.personal_best_position = np.copy(self.position)
        self.personal_best_value = np.full(self.initial_swarm_size, np.inf)
        self.global_best_position = np.zeros(dim)
        self.global_best_value = np.inf
        self.phi = 0.5
        self.w_max = 0.9
        self.w_min = 0.4
        self.c1_base = 2.0
        self.c2_base = 2.0
        self.learning_rate_decay = 0.99
        self.chaos_factor = 0.15
        self.levy_beta = 1.5
        self.dynamic_swarm = True

    def levy_flight(self, size):
        sigma1 = pow((1.0 + self.levy_beta) * np.math.gamma((1.0 + self.levy_beta) / 2.0) / 
                     (np.math.gamma(1.0 + self.levy_beta) * self.levy_beta * pow(2.0, ((self.levy_beta - 1.0) / 2.0))), 1.0 / self.levy_beta)
        u = np.random.normal(0, sigma1, size)
        v = np.random.normal(0, 1, size)
        step = u / (np.abs(v) ** (1.0 / self.levy_beta))
        return step

    def update_swarm_size(self, evals):
        if self.dynamic_swarm:
            new_size = max(5, int(self.initial_swarm_size * (1 - evals / self.budget)))
            if new_size < len(self.position):
                self.position = self.position[:new_size]
                self.velocity = self.velocity[:new_size]
                self.personal_best_position = self.personal_best_position[:new_size]
                self.personal_best_value = self.personal_best_value[:new_size]

    def __call__(self, func):
        evals = 0
        while evals < self.budget:
            self.update_swarm_size(evals)
            fitness_values = np.array([func(pos) for pos in self.position])
            evals += len(self.position)

            better_mask = fitness_values < self.personal_best_value
            self.personal_best_value[better_mask] = fitness_values[better_mask]
            self.personal_best_position[better_mask] = self.position[better_mask]

            min_fitness_idx = np.argmin(fitness_values)
            if fitness_values[min_fitness_idx] < self.global_best_value:
                self.global_best_value = fitness_values[min_fitness_idx]
                self.global_best_position = self.position[min_fitness_idx]

            inertia_weight = self.w_max - ((self.w_max - self.w_min) * (evals / self.budget))
            r1 = np.random.rand(len(self.position), self.dim)
            r2 = np.random.rand(len(self.position), self.dim)
            
            c1 = self.c1_base * (self.learning_rate_decay ** (evals / len(self.position)))
            c2 = self.c2_base * (self.learning_rate_decay ** (evals / len(self.position)))
            
            cognitive_component = c1 * r1 * (self.personal_best_position - self.position)
            social_component = c2 * r2 * (self.global_best_position - self.position)
            self.velocity = (inertia_weight * self.velocity + cognitive_component + social_component)
            
            chaos_component = self.chaos_factor * (np.random.rand(len(self.position), self.dim) - 0.5) * \
                              np.sin(1.5 * np.pi * evals / self.budget)
            new_position = self.position + self.velocity + chaos_component

            levy_component = 0.01 * self.levy_flight((len(self.position), self.dim))
            new_position += levy_component

            new_position = np.clip(new_position, -5.0, 5.0)
            self.position = new_position

        return self.global_best_position, self.global_best_value