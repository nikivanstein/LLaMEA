import numpy as np

class DynamicSwarmOptimizer:
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
        self.mutation_rate = 0.1

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

            # Dynamic adjustment of mutation rate
            self.mutation_rate = max(0.01, 0.2 * (1 - self.func_evaluations / self.budget))

            # Update velocities and positions
            for i in range(self.population_size):
                cognitive_component = np.random.random(self.dim) * (self.p_best_positions[i] - self.positions[i])
                social_component = np.random.random(self.dim) * (self.global_best_position - self.positions[i])
                inertia_component = 0.7 * self.velocities[i]

                self.velocities[i] = inertia_component + cognitive_component + social_component
                self.positions[i] += self.velocities[i]

                # Apply mutation
                mutation = np.random.uniform(-1, 1, self.dim) * self.mutation_rate
                self.positions[i] += mutation

                # Boundary check
                self.positions[i] = np.clip(self.positions[i], self.lower_bound, self.upper_bound)

        return self.global_best_position