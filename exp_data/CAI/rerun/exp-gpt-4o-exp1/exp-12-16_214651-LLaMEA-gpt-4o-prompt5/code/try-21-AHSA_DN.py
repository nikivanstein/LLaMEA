import numpy as np

class AHSA_DN:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.evaluations = 0
        self.best_position = None
        self.best_value = float('inf')

    def __call__(self, func):
        position = np.random.uniform(self.lower_bound, self.upper_bound, self.dim)
        value = func(position)
        self.evaluations += 1

        if value < self.best_value:
            self.best_value = value
            self.best_position = position

        hypercube_size = (self.upper_bound - self.lower_bound) / 4
        shrink_factor = 0.85
        success_rate_threshold = 0.2
        success_count = 0
        stagnation_counter = 0  # Initialize stagnation counter

        while self.evaluations < self.budget:
            for _ in range(max(1, int(self.dim / 2))):
                if self.evaluations >= self.budget:
                    break
                direction = np.random.uniform(-hypercube_size, hypercube_size * (1 + self.evaluations / self.budget), self.dim)
                candidate_position = position + direction
                candidate_position = np.clip(candidate_position, self.lower_bound, self.upper_bound)
                candidate_value = func(candidate_position)
                self.evaluations += 1

                if candidate_value < self.best_value:
                    self.best_value = candidate_value
                    self.best_position = candidate_position
                    position = candidate_position
                    hypercube_size *= shrink_factor
                    success_count += 1
                    stagnation_counter = 0  # Reset stagnation counter on improvement
                else:
                    stagnation_counter += 1  # Increase stagnation counter if no improvement

            if success_count / max(1, int(self.dim / 2)) > success_rate_threshold:
                shrink_factor *= 1.1
                hypercube_size /= shrink_factor * (1 + self.evaluations / self.budget)
            success_count = 0

            if stagnation_counter > 10:  # Trigger restart if stagnation persists
                position = np.random.uniform(self.lower_bound, self.upper_bound, self.dim)
                stagnation_counter = 0  # Reset stagnation counter

        return self.best_position, self.best_value