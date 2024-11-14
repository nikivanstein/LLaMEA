import numpy as np

class APSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 30
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population = np.random.uniform(self.lower_bound, self.upper_bound, (self.pop_size, self.dim))
        self.velocities = np.random.uniform(-1.0, 1.0, (self.pop_size, self.dim))
        self.personal_best_positions = np.copy(self.population)
        self.personal_best_fitness = np.full(self.pop_size, float('inf'))
        self.global_best_position = None
        self.global_best_fitness = float('inf')
        self.evaluations = 0
        self.w_max = 0.9
        self.w_min = 0.4
        self.c1 = 2.0
        self.c2 = 2.0

    def evaluate(self, func, solution):
        self.evaluations += 1
        return func(solution)

    def update_velocity(self, particle_idx, inertia_weight):
        r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
        cognitive_component = self.c1 * r1 * (self.personal_best_positions[particle_idx] - self.population[particle_idx])
        social_component = self.c2 * r2 * (self.global_best_position - self.population[particle_idx])
        self.velocities[particle_idx] = (inertia_weight * self.velocities[particle_idx] +
                                         cognitive_component + social_component)
        speed_limit = (self.upper_bound - self.lower_bound) * 0.1
        self.velocities[particle_idx] = np.clip(self.velocities[particle_idx], -speed_limit, speed_limit)

    def update_position(self, particle_idx):
        self.population[particle_idx] = self.population[particle_idx] + self.velocities[particle_idx]
        self.population[particle_idx] = np.clip(self.population[particle_idx], self.lower_bound, self.upper_bound)

    def __call__(self, func):
        np.random.seed(42)

        # Initial evaluation
        for i in range(self.pop_size):
            current_fitness = self.evaluate(func, self.population[i])
            self.personal_best_fitness[i] = current_fitness
            if current_fitness < self.global_best_fitness:
                self.global_best_fitness = current_fitness
                self.global_best_position = self.population[i]

        while self.evaluations < self.budget:
            inertia_weight = self.w_max - ((self.w_max - self.w_min) * (self.evaluations / self.budget))
            for i in range(self.pop_size):
                self.update_velocity(i, inertia_weight)
                self.update_position(i)

                current_fitness = self.evaluate(func, self.population[i])
                if current_fitness < self.personal_best_fitness[i]:
                    self.personal_best_fitness[i] = current_fitness
                    self.personal_best_positions[i] = self.population[i]

                if current_fitness < self.global_best_fitness:
                    self.global_best_fitness = current_fitness
                    self.global_best_position = self.population[i]

        return self.global_best_position