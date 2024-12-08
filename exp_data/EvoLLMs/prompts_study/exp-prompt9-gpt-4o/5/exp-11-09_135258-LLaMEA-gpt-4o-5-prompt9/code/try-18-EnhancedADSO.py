import numpy as np

class EnhancedADSO:
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
        sampling_rate = 0.12  # Adjusted initial sampling rate
        max_iterations = self.budget * 0.38  # Adjusted exploration budget to 38%
        shrink_factor = 0.91  # Slightly modified shrink factor
        learning_rate_adjustment = 1.02  # New adaptive learning rate adjustment

        while evals < self.budget:
            dynamic_candidates = 32 if exploration_phase else 14  # Slight adjustment in candidate number
            candidates = self._generate_candidates(best_solution, sampling_rate, dynamic_candidates)

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
                sampling_rate *= shrink_factor
                if evals >= max_iterations:
                    exploration_phase = False
                    convergence_threshold = np.std([func(c) for c in candidates])  # Adaptive convergence threshold
            else:
                sampling_rate *= (learning_rate_adjustment * shrink_factor)  # Modified exploitation scaling

        return best_solution

    def _generate_candidates(self, center, rate, num_samples):
        candidates = []
        for _ in range(num_samples):
            perturbation = np.random.normal(0, 1, self.dim) * rate
            candidate = np.clip(center + perturbation, self.lower_bound, self.upper_bound)
            candidates.append(candidate)
        return candidates