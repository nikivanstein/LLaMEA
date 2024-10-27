import numpy as np

class DynamicPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.inertia_weight_max = 0.9
        self.inertia_weight_min = 0.4
        self.social_weight = 0.5
        self.cognitive_weight = 0.5
        self.population_size = 20
        self.velocity_max = 0.2 * (self.upper_bound - self.lower_bound)
        self.position = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        self.velocity = np.zeros((self.population_size, self.dim))
        self.personal_best_position = self.position.copy()
        self.personal_best_fitness = np.array([float('inf')] * self.population_size)
        self.global_best_position = np.zeros(self.dim)
        self.global_best_fitness = float('inf')

    def dynamic_inertia_weight(self, iter_num, max_iter):
        return self.inertia_weight_max - (iter_num / max_iter) * (self.inertia_weight_max - self.inertia_weight_min)

    def update_velocity(self):
        inertia_weight = self.dynamic_inertia_weight(iter_num, self.budget)
        cognitive_component = self.cognitive_weight * np.random.rand() * (self.personal_best_position - self.position)
        social_component = self.social_weight * np.random.rand() * (self.global_best_position - self.position)
        self.velocity = inertia_weight * self.velocity + cognitive_component + social_component
        self.velocity = np.clip(self.velocity, -self.velocity_max, self.velocity_max)

    def __call__(self, func):
        for iter_num in range(self.budget):
            fitness = np.array([func(x) for x in self.position])

            for i in range(self.population_size):
                if fitness[i] < self.personal_best_fitness[i]:
                    self.personal_best_fitness[i] = fitness[i]
                    self.personal_best_position[i] = self.position[i]
                
                if fitness[i] < self.global_best_fitness:
                    self.global_best_fitness = fitness[i]
                    self.global_best_position = self.position[i]

            self.update_velocity()
            self.position += self.velocity
            self.position = np.clip(self.position, self.lower_bound, self.upper_bound)

        return self.global_best_position