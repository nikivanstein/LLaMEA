import numpy as np

class ADSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0

    def __call__(self, func):
        # Initialize parameters
        evals = 0
        best_solution = np.random.uniform(self.lower_bound, self.upper_bound, self.dim)
        best_value = func(best_solution)
        evals += 1
        
        # Adaptive sampling parameters
        exploration_phase = True
        sampling_rate = 0.1
        max_iterations = self.budget // 2  # Split budget into exploration and exploitation

        while evals < self.budget:
            if exploration_phase:
                candidates = self._generate_candidates(best_solution, sampling_rate, 30)
            else:
                candidates = self._generate_candidates(best_solution, sampling_rate / 2, 10)

            for candidate in candidates:
                if evals >= self.budget:
                    break
                candidate_value = func(candidate)
                evals += 1
                if candidate_value < best_value:
                    best_value = candidate_value
                    best_solution = candidate

            # Update exploration-exploitation balance
            if exploration_phase:
                sampling_rate *= 0.85
                if evals >= max_iterations:
                    exploration_phase = False
            else:
                sampling_rate *= 1.05

        return best_solution

    def _generate_candidates(self, center, rate, num_samples):
        candidates = []
        for _ in range(num_samples):
            perturbation = np.random.uniform(-1, 1, self.dim) * rate
            candidate = np.clip(center + perturbation, self.lower_bound, self.upper_bound)
            candidates.append(candidate)
        return candidates