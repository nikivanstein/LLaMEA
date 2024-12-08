import numpy as np

class APSO_DT:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 30
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population = np.random.uniform(self.lower_bound, self.upper_bound, (self.pop_size, self.dim))
        self.velocities = np.random.uniform(-1, 1, (self.pop_size, self.dim))
        self.personal_best_positions = np.copy(self.population)
        self.personal_best_fitness = np.full(self.pop_size, float('inf'))
        self.global_best_position = None
        self.global_best_fitness = float('inf')
        self.evaluations = 0
        self.inertia_weight = 0.729
        self.cognitive_constant = 1.49445
        self.social_constant = 1.49445

    def evaluate(self, func, solution):
        self.evaluations += 1
        return func(solution)

    def update_topology(self):
        # Dynamic neighborhood connectivity
        np.random.shuffle(self.population)
        for i in range(self.pop_size):
            for j in range(i + 1, self.pop_size):
                if np.random.rand() < 0.2:  # 20% chance to connect
                    if self.personal_best_fitness[j] < self.personal_best_fitness[i]:
                        self.personal_best_positions[i] = self.personal_best_positions[j]
                        self.personal_best_fitness[i] = self.personal_best_fitness[j]

    def __call__(self, func):
        np.random.seed(42)

        # Initial evaluation
        for i in range(self.pop_size):
            fitness = self.evaluate(func, self.population[i])
            if fitness < self.personal_best_fitness[i]:
                self.personal_best_fitness[i] = fitness
                self.personal_best_positions[i] = self.population[i]

            if fitness < self.global_best_fitness:
                self.global_best_fitness = fitness
                self.global_best_position = self.population[i]

        while self.evaluations < self.budget:
            self.update_topology()

            for i in range(self.pop_size):
                r1 = np.random.rand(self.dim)
                r2 = np.random.rand(self.dim)
                cognitive_velocity = self.cognitive_constant * r1 * (self.personal_best_positions[i] - self.population[i])
                social_velocity = self.social_constant * r2 * (self.global_best_position - self.population[i])
                self.velocities[i] = (self.inertia_weight * self.velocities[i] +
                                      cognitive_velocity + social_velocity)
                self.population[i] += self.velocities[i]
                self.population[i] = np.clip(self.population[i], self.lower_bound, self.upper_bound)

                fitness = self.evaluate(func, self.population[i])
                if fitness < self.personal_best_fitness[i]:
                    self.personal_best_fitness[i] = fitness
                    self.personal_best_positions[i] = self.population[i]

                if fitness < self.global_best_fitness:
                    self.global_best_fitness = fitness
                    self.global_best_position = self.population[i]

        return self.global_best_position