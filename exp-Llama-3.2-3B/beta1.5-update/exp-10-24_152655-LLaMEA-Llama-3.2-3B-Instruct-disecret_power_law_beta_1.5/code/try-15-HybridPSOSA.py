import numpy as np
import random
import operator

class HybridPSOSA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.bounds = np.array([-5.0, 5.0]) * np.ones((dim, 1))
        self.f_best = np.inf
        self.x_best = np.zeros((dim, 1))
        self.f_evals = 0
        self.f_evals_best = 0
        self.swarm_size = 20
        self.pso_alpha = 0.18
        self.pso_beta = 0.18
        self.pso_gamma = 0.18
        self.sa_temp = 1000
        self.sa_alpha = 0.99

    def __call__(self, func):
        if self.f_evals >= self.budget:
            return self.x_best

        for _ in range(self.budget - self.f_evals):
            # Initialize a list of random candidates
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

            # Schedule the best candidate
            candidates = np.delete(candidates, np.where(candidates == best_candidate), axis=0)

            # Update the bounds
            self.bounds = np.array([np.min(candidates, axis=0), np.max(candidates, axis=0)])

            # PSO update
            for i in range(self.swarm_size):
                # Randomly select a candidate
                candidate = candidates[i % self.swarm_size]

                # Update velocity
                velocity = np.random.uniform(-1, 1, size=self.dim)
                candidate += velocity

                # Evaluate the candidate
                f_candidate = func(candidate)

                # Update the best candidate
                if f_candidate < f_evals_best:
                    f_evals_best = f_candidate
                    x_best = candidate

                # Update the best candidate in the swarm
                if f_candidate < f_candidates[i % self.swarm_size]:
                    f_candidates[i % self.swarm_size] = f_candidate
                    candidates[i % self.swarm_size] = candidate

                # Update the bounds
                self.bounds = np.array([np.min(candidates, axis=0), np.max(candidates, axis=0)])

            # SA update
            if self.f_evals % 10 == 0:
                # Generate a new candidate
                candidate = np.random.uniform(self.bounds[:, 0], self.bounds[:, 1], size=self.dim)

                # Evaluate the candidate
                f_candidate = func(candidate)

                # Calculate the probability of accepting the new candidate
                prob = np.exp((f_candidate - f_evals_best) / self.sa_temp)

                # Accept the new candidate with the calculated probability
                if random.random() < prob:
                    f_evals_best = f_candidate
                    x_best = candidate

        return self.x_best

# Example usage:
def func(x):
    return x[0]**2 + x[1]**2

hybrid_pso_soa = HybridPSOSA(budget=10, dim=2)
x_opt = hybrid_pso_soa(func)
print(x_opt)