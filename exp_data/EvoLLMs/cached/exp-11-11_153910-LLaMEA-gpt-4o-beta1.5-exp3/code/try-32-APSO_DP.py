import numpy as np

class APSO_DP:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 30
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population = np.random.uniform(self.lower_bound, self.upper_bound, (self.pop_size, self.dim))
        self.velocities = np.zeros((self.pop_size, self.dim))
        self.personal_best_positions = self.population.copy()
        self.personal_best_fitness = np.full(self.pop_size, float('inf'))
        self.best_global_position = None
        self.best_global_fitness = float('inf')
        self.evaluations = 0
        self.w = 0.7   # Inertia weight
        self.c1 = 1.5  # Cognitive coefficient
        self.c2 = 1.5  # Social coefficient
        self.F = 0.5   # Differential mutation factor

    def evaluate(self, func, solution):
        self.evaluations += 1
        return func(solution)

    def __call__(self, func):
        np.random.seed(42)

        # Initial evaluation
        for i in range(self.pop_size):
            fitness = self.evaluate(func, self.population[i])
            if fitness < self.personal_best_fitness[i]:
                self.personal_best_fitness[i] = fitness
                self.personal_best_positions[i] = self.population[i]
            if fitness < self.best_global_fitness:
                self.best_global_fitness = fitness
                self.best_global_position = self.population[i]

        while self.evaluations < self.budget:
            for i in range(self.pop_size):
                r1, r2 = np.random.rand(2)
                cognitive_component = self.c1 * r1 * (self.personal_best_positions[i] - self.population[i])
                social_component = self.c2 * r2 * (self.best_global_position - self.population[i])
                self.velocities[i] = self.w * self.velocities[i] + cognitive_component + social_component

                # Differential perturbation
                indices = [idx for idx in range(self.pop_size) if idx != i]
                x1, x2 = self.population[np.random.choice(indices, 2, replace=False)]
                perturbation = self.F * (x1 - x2)
                trial_position = self.population[i] + self.velocities[i] + perturbation
                trial_position = np.clip(trial_position, self.lower_bound, self.upper_bound)

                # Evaluate trial position
                trial_fitness = self.evaluate(func, trial_position)
                if trial_fitness < self.personal_best_fitness[i]:
                    self.personal_best_fitness[i] = trial_fitness
                    self.personal_best_positions[i] = trial_position
                if trial_fitness < self.best_global_fitness:
                    self.best_global_fitness = trial_fitness
                    self.best_global_position = trial_position

                # Update particle position
                self.population[i] = trial_position

        return self.best_global_position