import numpy as np

class HybridGradientPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = int(10 + 2 * np.sqrt(dim))
        self.positions = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, dim))
        self.velocities = np.random.uniform(-1, 1, (self.population_size, dim))
        self.p_best_positions = np.copy(self.positions)
        self.global_best_position = np.copy(self.positions[0])
        self.func_evaluations = 0
        self.best_score = float('inf')
        self.c1 = 1.5  # cognitive coefficient
        self.c2 = 1.5  # social coefficient
        self.inertia_weight = 0.7

    def estimate_gradient(self, func, position):
        epsilon = 1e-5
        grad = np.zeros(self.dim)
        for i in range(self.dim):
            step = np.zeros(self.dim)
            step[i] = epsilon
            grad[i] = (func(position + step) - func(position - step)) / (2 * epsilon)
            self.func_evaluations += 2
        return grad

    def __call__(self, func):
        while self.func_evaluations < self.budget:
            # Evaluate current population
            scores = np.array([func(pos) for pos in self.positions])
            self.func_evaluations += self.population_size

            # Update personal bests
            better_p_best_mask = scores < np.array([func(p) for p in self.p_best_positions])
            self.p_best_positions[better_p_best_mask] = self.positions[better_p_best_mask]

            # Update global best
            min_index = np.argmin(scores)
            if scores[min_index] < self.best_score:
                self.global_best_position = self.positions[min_index]
                self.best_score = scores[min_index]

            for i in range(self.population_size):
                # Estimate gradient and use it to adjust position
                gradient = self.estimate_gradient(func, self.positions[i])
                gradient_step = -0.01 * gradient  # step size for gradient adjustment

                # Particle swarm update
                cognitive_component = self.c1 * np.random.random(self.dim) * (self.p_best_positions[i] - self.positions[i])
                social_component = self.c2 * np.random.random(self.dim) * (self.global_best_position - self.positions[i])
                self.velocities[i] = self.inertia_weight * self.velocities[i] + cognitive_component + social_component
                self.positions[i] += self.velocities[i] + gradient_step

                # Boundary check
                self.positions[i] = np.clip(self.positions[i], self.lower_bound, self.upper_bound)

            if self.func_evaluations >= self.budget:
                break

        return self.global_best_position