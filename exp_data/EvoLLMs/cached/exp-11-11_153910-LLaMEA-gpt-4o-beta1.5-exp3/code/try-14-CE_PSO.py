import numpy as np

class CE_PSO:
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
        self.w = 0.9  # Inertia weight
        self.c1 = 2.0 # Cognitive component
        self.c2 = 2.0 # Social component
        self.chao_seq = self.init_chaotic_map()

    def init_chaotic_map(self):
        x0 = np.random.rand()
        return [x0]

    def update_chaotic_map(self):
        x_next = 4 * self.chao_seq[-1] * (1 - self.chao_seq[-1])
        self.chao_seq.append(x_next)
        # Use chaotic sequence to adjust w, c1, c2
        self.w = 0.5 + 0.4 * x_next
        self.c1 = 1.5 + 1.0 * x_next
        self.c2 = 1.5 + 1.0 * (1 - x_next)

    def evaluate(self, func, solution):
        self.evaluations += 1
        return func(solution)

    def __call__(self, func):
        np.random.seed(42)
        
        # Initial evaluation
        for i in range(self.pop_size):
            fitness = self.evaluate(func, self.population[i])
            self.personal_best_fitness[i] = fitness
            if fitness < self.global_best_fitness:
                self.global_best_fitness = fitness
                self.global_best_position = self.population[i]

        while self.evaluations < self.budget:
            self.update_chaotic_map()
            for i in range(self.pop_size):
                r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
                cognitive = self.c1 * r1 * (self.personal_best_positions[i] - self.population[i])
                social = self.c2 * r2 * (self.global_best_position - self.population[i])
                self.velocities[i] = self.w * self.velocities[i] + cognitive + social
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