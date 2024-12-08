import numpy as np

class AdaptiveRandomLocalSearch:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.history = []

    def __call__(self, func):
        evals = 0
        best_solution = np.random.uniform(self.lower_bound, self.upper_bound, self.dim)
        best_value = func(best_solution)
        evals += 1

        # Store the initial solution
        self.history.append((best_solution, best_value))

        while evals < self.budget:
            # Calculate adaptation factor based on feedback
            adaptation_factor = self.calculate_adaptation()

            # Generate candidate solutions around the current best
            candidate_solutions = self.generate_candidates(best_solution, adaptation_factor, evals)
            candidate_values = [func(cand) for cand in candidate_solutions]
            evals += len(candidate_solutions)

            # Find the best candidate
            best_idx = np.argmin(candidate_values)
            if candidate_values[best_idx] < best_value:
                best_value = candidate_values[best_idx]
                best_solution = candidate_solutions[best_idx]

            # Store the new best solution
            self.history.append((best_solution, best_value))

        return best_solution

    def generate_candidates(self, current_best, adaptation_factor, evals):
        num_candidates = min(self.budget - len(self.history), 10)
        # Modified perturbation_radius calculation incorporating adaptive scaling
        adaptive_scale = 1.0 - (evals / self.budget)  # Adaptive scaling based on iteration
        # Modify perturbation_radius to include Gaussian component
        perturbation_radius = adaptation_factor * adaptive_scale * (self.upper_bound - self.lower_bound) / (
            10 * adaptation_factor * np.abs(np.random.standard_cauchy()) + np.random.normal(0, 0.1)
        )
        candidates = [
            np.clip(current_best + np.random.uniform(-perturbation_radius, perturbation_radius, self.dim),
                    self.lower_bound, self.upper_bound)
            for _ in range(num_candidates)
        ]
        return candidates

    def calculate_adaptation(self):
        # Use past history to adaptively adjust the exploration range
        if len(self.history) < 2:
            return 1.0  # Start with maximum exploration
        else:
            # Calculate success rate in recent history
            recent_history = self.history[-10:]  # Look at the last 10 evaluations
            improvements = sum(1 for i in range(1, len(recent_history))
                               if recent_history[i][1] < recent_history[i - 1][1])
            success_rate = improvements / len(recent_history)
            return max(0.1, min(1.0, 1.0 - 0.5 * success_rate))