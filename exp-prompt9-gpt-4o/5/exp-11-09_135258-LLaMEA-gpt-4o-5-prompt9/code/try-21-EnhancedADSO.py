import numpy as np

class EnhancedADSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0

    def __call__(self, func):
        evals = 0
        best_solution = np.random.uniform(self.lower_bound, self.upper_bound, self.dim)
        best_value = func(best_solution)
        evals += 1
        
        exploration_phase = True
        sampling_rate = 0.12
        max_iterations = self.budget * 0.35  # Reduced exploration budget
        shrink_factor = 0.92  # Slightly reduced shrink factor

        while evals < self.budget:
            dynamic_candidates = 40 if exploration_phase else 8  # Enhanced candidate diversity
            candidates = self._generate_candidates(best_solution, sampling_rate, dynamic_candidates)

            for candidate in candidates:
                if evals >= self.budget:
                    break
                candidate_value = func(candidate)
                evals += 1
                if candidate_value < best_value:
                    best_value = candidate_value
                    best_solution = candidate

            if exploration_phase:
                sampling_rate *= shrink_factor
                if evals >= max_iterations:
                    exploration_phase = False
                    convergence_threshold = np.mean([abs(func(c) - best_value) for c in candidates])
            else:
                sampling_rate *= (1.05 * shrink_factor)  # Fine-tuned exploitation scaling

        return best_solution

    def _generate_candidates(self, center, rate, num_samples):
        candidates = []
        for _ in range(num_samples):
            perturbation = np.random.normal(0, 1, self.dim) * rate
            candidate = np.clip(center + perturbation, self.lower_bound, self.upper_bound)
            candidates.append(candidate)
        return candidates