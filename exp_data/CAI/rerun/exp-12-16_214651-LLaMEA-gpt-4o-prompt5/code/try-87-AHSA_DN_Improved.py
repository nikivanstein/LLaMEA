import numpy as np

class AHSA_DN_Improved:
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

        hypercube_size = (self.upper_bound - self.lower_bound) / (2.5 * np.log1p(self.dim))
        shrink_factor = 0.9
        success_rate_threshold = 0.3
        success_count = 0
        dynamic_mutation_rate = 0.1

        last_direction = np.zeros(self.dim)

        initial_temp = 1.0
        temperature = initial_temp

        while self.evaluations < self.budget:
            for _ in range(max(1, int(self.dim / 2))):
                if self.evaluations >= self.budget:
                    break
                direction = np.random.uniform(-hypercube_size, hypercube_size, self.dim)
                direction *= np.abs(np.cos(self.evaluations / self.budget * np.pi))
                direction *= (1 + np.random.rand() * dynamic_mutation_rate)
                direction += 0.3 * last_direction
                candidate_position = position + direction
                candidate_position = np.clip(candidate_position, self.lower_bound, self.upper_bound)
                candidate_value = func(candidate_position)
                self.evaluations += 1

                if candidate_value < self.best_value or np.exp((value - candidate_value) / temperature) > np.random.rand():
                    self.best_value = candidate_value
                    self.best_position = candidate_position
                    position = candidate_position
                    hypercube_size *= shrink_factor
                    if candidate_value < value:
                        dynamic_mutation_rate *= 0.88
                    success_count += 1
                last_direction = direction * (0.99 + 0.005 * np.random.rand())

            temperature *= 0.95
            if success_count / max(1, int(self.dim / 2)) > success_rate_threshold:
                shrink_factor *= 1.03
                hypercube_size /= shrink_factor * (1 + np.tanh(self.evaluations / (self.budget * 1.2)))
            success_count = 0

        return self.best_position, self.best_value