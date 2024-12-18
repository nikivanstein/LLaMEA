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
        # Initial random position
        position = np.random.uniform(self.lower_bound, self.upper_bound, self.dim)
        value = func(position)
        self.evaluations += 1

        # Save the best found so far
        if value < self.best_value:
            self.best_value = value
            self.best_position = position

        # Define hypercube initial size and step factor
        hypercube_size = (self.upper_bound - self.lower_bound) / 3.5
        shrink_factor = 0.9  # Line changed
        success_rate_threshold = 0.3  # Line changed
        success_count = 0

        while self.evaluations < self.budget:
            # Generate new candidates within the hypercube
            for _ in range(max(1, int(self.dim / 2))):
                if self.evaluations >= self.budget:
                    break
                # Randomly pick a direction and step size within the hypercube
                direction = np.random.uniform(-hypercube_size, hypercube_size, self.dim)
                direction *= np.abs(np.cos(self.evaluations / self.budget * np.pi))
                adaptive_rate = 0.5 + 0.5 * (self.evaluations / self.budget)  # Line changed
                candidate_position = position + direction * adaptive_rate  # Line changed
                # Clamping within bounds
                candidate_position = np.clip(candidate_position, self.lower_bound, self.upper_bound)
                candidate_value = func(candidate_position)
                self.evaluations += 1

                # Update the best found so far
                if candidate_value < self.best_value:
                    self.best_value = candidate_value
                    self.best_position = candidate_position
                    position = candidate_position  # Move towards better position
                    hypercube_size *= shrink_factor  # Shrink hypercube for finer search
                    success_count += 1

            # Adjust hypercube size based on success rate and evaluation progress
            if success_count / max(1, int(self.dim / 2)) > success_rate_threshold:
                shrink_factor *= 1.02  # Line changed
                hypercube_size /= shrink_factor * (1 + np.tanh(self.evaluations / self.budget))
            success_count = 0  # Reset for next round

        return self.best_position, self.best_value