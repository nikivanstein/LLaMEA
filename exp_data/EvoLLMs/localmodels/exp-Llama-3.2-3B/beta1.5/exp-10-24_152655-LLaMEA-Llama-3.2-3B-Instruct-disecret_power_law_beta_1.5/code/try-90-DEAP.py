import numpy as np
import random
import operator

class DEAP:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.bounds = np.array([-5.0, 5.0]) * np.ones((dim, 1))
        self.f_best = np.inf
        self.x_best = np.zeros((dim, 1))
        self.f_evals = 0
        self.f_evals_best = 0
        self.p = 0.018518518518518517

    def __call__(self, func):
        if self.f_evals >= self.budget:
            return self.x_best

        for _ in range(self.budget - self.f_evals):
            # Generate three random candidates
            candidates = np.random.uniform(self.bounds[:, 0], self.bounds[:, 1], size=(self.dim, 1))

            # Evaluate the candidates
            f_candidates = func(candidates)

            # Update the best solution
            f_evals = f_candidates[0]
            x_best = candidates[0]
            f_evals_best = f_evals

            # Update the best solution if necessary
            if f_evals < self.f_best:
                self.f_best = f_evals
                self.x_best = x_best
                self.f_evals_best = f_evals

            # Select the best candidate
            best_candidate = candidates[np.argmin(f_candidates)]

            # Calculate the differential evolution parameters
            diff1 = candidates[1] - best_candidate
            diff2 = candidates[2] - best_candidate
            diff3 = candidates[0] - best_candidate

            # Calculate the trial vector
            trial_vector = best_candidate + self.p * np.array([diff1, diff2, diff3])

            # Evaluate the trial vector
            f_trial = func(trial_vector)

            # Update the bounds
            self.bounds = np.array([np.min([candidates[0], trial_vector, best_candidate]), np.max([candidates[0], trial_vector, best_candidate])])

            # Update the best solution if necessary
            if f_trial < f_evals_best:
                self.f_evals_best = f_trial
                self.x_best = trial_vector

        return self.x_best

# Example usage:
def func(x):
    return x[0]**2 + x[1]**2

DEAP_obj = DEAP(budget=10, dim=2)
x_opt = DEAP_obj(func)
print(x_opt)