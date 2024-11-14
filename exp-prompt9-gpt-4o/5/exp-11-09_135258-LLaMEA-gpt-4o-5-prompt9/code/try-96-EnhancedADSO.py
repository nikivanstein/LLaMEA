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
        sampling_rate = 0.12  # Slightly adjusted for enhanced initial exploration
        adaptive_rate = 0.16  # Increased for more dynamic adaptation
        max_iterations = self.budget * 0.2  # Adjusted to speed up phase transition
        shrink_factor = 0.93  # Modified shrink factor for faster convergence
        chaotic_sequence = self._generate_chaotic_sequence(int(self.budget))

        while evals < self.budget:
            dynamic_candidates = 22 if exploration_phase else 17  # Adjusted candidate size
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
            else:
                sampling_rate = max(sampling_rate * (1.1 * shrink_factor), adaptive_rate)

            # Introduce a random walk effect for diversification
            if evals % 18 == 0:  # Adjusted interval for random walk
                random_walk = np.random.uniform(self.lower_bound, self.upper_bound, self.dim)
                candidates.append(np.clip(random_walk, self.lower_bound, self.upper_bound))

            sampling_rate *= (1 + 0.015 * chaotic_sequence[evals % len(chaotic_sequence)])  # Increased chaotic influence

        return best_solution

    def _generate_candidates(self, center, rate, num_samples):
        candidates = []
        for _ in range(num_samples):
            perturbation = self._levy_flight(self.dim) * rate
            candidate = np.clip(center + perturbation, self.lower_bound, self.upper_bound)
            candidates.append(candidate)
        return candidates

    def _generate_chaotic_sequence(self, length):
        sequence = np.zeros(length)
        sequence[0] = 0.65  # Changed initial value for chaotic sequence
        for i in range(1, length):
            sequence[i] = 4.0 * sequence[i-1] * (1 - sequence[i-1])
        return sequence

    def _levy_flight(self, dim, beta=1.5):
        sigma = (np.math.gamma(1 + beta) * np.sin(np.pi * beta / 2) /
                 (np.math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (1 / beta)
        u = np.random.normal(0, sigma, dim)
        v = np.random.normal(0, 1, dim)
        step = u / np.abs(v) ** (1 / beta)
        return step