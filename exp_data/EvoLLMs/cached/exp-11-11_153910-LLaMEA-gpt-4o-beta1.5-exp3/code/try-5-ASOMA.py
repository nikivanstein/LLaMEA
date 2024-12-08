import numpy as np

class ASOMA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 30
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population = np.random.uniform(self.lower_bound, self.upper_bound, (self.pop_size, self.dim))
        self.velocities = np.random.uniform(-1, 1, (self.pop_size, self.dim))
        self.best_personal_positions = np.copy(self.population)
        self.best_personal_fitness = np.full(self.pop_size, float('inf'))
        self.best_global_position = None
        self.best_global_fitness = float('inf')
        self.evaluations = 0

    def evaluate(self, func, solution):
        self.evaluations += 1
        return func(solution)

    def migrate(self):
        # Migrate individuals to new random positions
        for i in range(self.pop_size):
            if np.random.rand() < 0.1:  # Probability of migration
                self.population[i] = np.random.uniform(self.lower_bound, self.upper_bound, self.dim)
                self.velocities[i] = np.random.uniform(-1, 1, self.dim)

    def __call__(self, func):
        np.random.seed(42)
        w_max, w_min = 0.9, 0.4
        c1, c2 = 2.0, 2.0
        v_max = (self.upper_bound - self.lower_bound) * 0.1

        while self.evaluations < self.budget:
            inertia_weight = w_max - (w_max - w_min) * (self.evaluations / self.budget)

            for i in range(self.pop_size):
                fitness = self.evaluate(func, self.population[i])
                
                if fitness < self.best_personal_fitness[i]:
                    self.best_personal_fitness[i] = fitness
                    self.best_personal_positions[i] = self.population[i]

                if fitness < self.best_global_fitness:
                    self.best_global_fitness = fitness
                    self.best_global_position = self.population[i]

            for i in range(self.pop_size):
                r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
                cognitive_component = c1 * r1 * (self.best_personal_positions[i] - self.population[i])
                social_component = c2 * r2 * (self.best_global_position - self.population[i])
                self.velocities[i] = inertia_weight * self.velocities[i] + cognitive_component + social_component
                
                # Clamp velocities
                self.velocities[i] = np.clip(self.velocities[i], -v_max, v_max)
                
                # Update positions
                self.population[i] = np.clip(self.population[i] + self.velocities[i], self.lower_bound, self.upper_bound)

            # Apply migration strategy
            self.migrate()
        
        return self.best_global_position