import numpy as np

class ImprovedHybridAQPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.swarm_size = 50 + (dim * 4)
        self.position = np.random.uniform(-5.0, 5.0, (self.swarm_size, dim))
        self.velocity = np.zeros((self.swarm_size, dim))
        self.personal_best_position = np.copy(self.position)
        self.personal_best_value = np.full(self.swarm_size, np.inf)
        self.global_best_position = np.zeros(dim)
        self.global_best_value = np.inf
        self.w = 0.7
        self.c1_init = 1.5
        self.c2_init = 1.5
        self.c1_end = 2.5
        self.c2_end = 2.5
        self.adaptive_levy_rate = 0.95
        self.levy_beta = 1.5

    def levy_flight(self, size):
        sigma1 = pow((1.0 + self.levy_beta) * np.math.gamma((1.0 + self.levy_beta) / 2.0) / 
                     (np.math.gamma(1.0 + self.levy_beta) * self.levy_beta * pow(2.0, ((self.levy_beta - 1.0) / 2.0))), 1.0 / self.levy_beta)
        u = np.random.normal(0, sigma1, size)
        v = np.random.normal(0, 1, size)
        step = u / (np.abs(v) ** (1.0 / self.levy_beta))
        return step

    def dynamic_leader_selection(self, fitness_values):
        probabilities = np.exp(-fitness_values / np.sum(fitness_values))
        idx = np.random.choice(np.arange(self.swarm_size), p=probabilities / probabilities.sum())
        return idx

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

            inertia_weight = self.w
            c1 = self.c1_init + (self.c1_end - self.c1_init) * (evals / self.budget)
            c2 = self.c2_init + (self.c2_end - self.c2_init) * (evals / self.budget)

            r1 = np.random.rand(self.swarm_size, self.dim)
            r2 = np.random.rand(self.swarm_size, self.dim)
            
            leader_index = self.dynamic_leader_selection(fitness_values)
            dynamic_leader_position = self.position[leader_index]
            
            cognitive_component = c1 * r1 * (self.personal_best_position - self.position)
            social_component = c2 * r2 * (dynamic_leader_position - self.position)
            self.velocity = (inertia_weight * self.velocity + cognitive_component + social_component)
            
            new_position = self.position + self.velocity
            if np.random.rand() < self.adaptive_levy_rate:
                levy_component = 0.01 * self.levy_flight((self.swarm_size, self.dim))
                new_position += levy_component

            new_position = np.clip(new_position, -5.0, 5.0)
            self.position = new_position

        return self.global_best_position, self.global_best_value