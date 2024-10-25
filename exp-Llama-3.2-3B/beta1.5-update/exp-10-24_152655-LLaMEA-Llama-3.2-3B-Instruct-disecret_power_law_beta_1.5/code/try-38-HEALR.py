import numpy as np
import random
import operator

class HEALR:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.bounds = np.array([-5.0, 5.0]) * np.ones((dim, 1))
        self.f_best = np.inf
        self.x_best = np.zeros((dim, 1))
        self.f_evals = 0
        self.f_evals_best = 0
        self.lr = 0.1  # initial learning rate

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

            # Adaptive learning rate update
            if self.f_evals / self.budget < 0.2:
                self.lr *= 0.9  # decrease learning rate
            else:
                self.lr *= 1.1  # increase learning rate

            # Perform evolutionary search with adaptive learning rate
            for _ in range(int(self.lr)):
                # Generate a new candidate using mutation and crossover
                new_candidate = np.random.uniform(self.bounds[:, 0], self.bounds[:, 1], size=(self.dim, 1))
                new_candidate = self.crossover(new_candidate, best_candidate)
                new_candidate = self.mutate(new_candidate)

                # Evaluate the new candidate
                f_new = func(new_candidate)

                # Update the best solution if necessary
                if f_new < f_evals_best:
                    f_evals_best = f_new
                    x_best = new_candidate

        return self.x_best

    def crossover(self, parent1, parent2):
        # Perform single-point crossover
        crossover_point = random.randint(1, self.dim - 1)
        child = np.concatenate((parent1[:crossover_point], parent2[crossover_point:]))
        return child

    def mutate(self, individual):
        # Perform Gaussian mutation
        mutation_std = 0.1 * (self.f_evals / self.budget)
        mutated_individual = individual + np.random.normal(0, mutation_std, size=(self.dim, 1))
        return mutated_individual

# Example usage:
def func(x):
    return x[0]**2 + x[1]**2

healr = HEALR(budget=10, dim=2)
x_opt = healr(func)
print(x_opt)