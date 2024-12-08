import numpy as np

class GradientDrivenAdaptiveParticle:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = int(10 + 2 * np.sqrt(dim))
        self.positions = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, dim))
        self.velocities = np.random.uniform(-1, 1, (self.population_size, dim))
        self.best_positions = np.copy(self.positions)
        self.global_best_position = np.copy(self.positions[0])
        self.func_evaluations = 0
        self.best_score = float('inf')
        self.learning_rate = 0.1

    def __call__(self, func):
        while self.func_evaluations < self.budget:
            # Evaluate current population
            scores = np.array([func(pos) for pos in self.positions])
            self.func_evaluations += self.population_size

            # Update personal and global bests
            for i in range(self.population_size):
                if scores[i] < func(self.best_positions[i]):
                    self.best_positions[i] = self.positions[i]
                if scores[i] < self.best_score:
                    self.global_best_position = self.positions[i]
                    self.best_score = scores[i]

            # Adaptive learning rate adjustment
            self.learning_rate = 0.1 * (1 - self.func_evaluations / self.budget)

            # Estimate gradient and update velocities and positions
            for i in range(self.population_size):
                gradient_approx = np.zeros(self.dim)
                epsilon = 1e-8
                for d in range(self.dim):
                    step = np.zeros(self.dim)
                    step[d] = epsilon
                    gradient_approx[d] = (func(self.positions[i] + step) - func(self.positions[i] - step)) / (2 * epsilon)
                    self.func_evaluations += 2

                inertia_component = 0.5 * self.velocities[i]
                cognitive_component = np.random.random(self.dim) * (self.best_positions[i] - self.positions[i])
                social_component = np.random.random(self.dim) * (self.global_best_position - self.positions[i])
                gradient_component = -self.learning_rate * gradient_approx

                self.velocities[i] = inertia_component + cognitive_component + social_component + gradient_component
                self.positions[i] += self.velocities[i]

                # Boundary check
                self.positions[i] = np.clip(self.positions[i], self.lower_bound, self.upper_bound)

        return self.global_best_position