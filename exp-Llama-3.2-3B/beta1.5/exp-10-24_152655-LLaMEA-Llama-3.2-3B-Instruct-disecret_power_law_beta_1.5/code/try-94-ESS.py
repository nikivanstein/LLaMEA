import numpy as np
import random
import operator

class ESS:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.bounds = np.array([-5.0, 5.0]) * np.ones((dim, 1))
        self.f_best = np.inf
        self.x_best = np.zeros((dim, 1))
        self.f_evals = 0
        self.f_evals_best = 0
        self.population = []

    def __call__(self, func):
        if self.f_evals >= self.budget:
            return self.x_best

        # Initialize the population with random candidates
        for _ in range(self.budget):
            candidate = np.random.uniform(self.bounds[:, 0], self.bounds[:, 1], size=(self.dim, 1))
            self.population.append(candidate)

        # Evaluate the candidates
        f_candidates = [func(candidate) for candidate in self.population]

        # Update the best solution
        f_evals = min(f_candidates)
        x_best = self.population[f_candidates.index(f_evals)]
        f_evals_best = f_evals

        # Update the best solution if necessary
        if f_evals < self.f_best:
            self.f_best = f_evals
            self.x_best = x_best
            self.f_evals_best = f_evals

        # Select the best candidate with probability-based selection
        selected = []
        for _ in range(self.budget):
            if len(selected) < self.budget:
                selected.append(self.population[np.random.choice(len(self.population), p=[1.0 / len(self.population) for _ in range(len(self.population))])])
            else:
                selected.remove(self.population[np.random.choice(len(self.population), p=[1.0 / len(self.population) for _ in range(len(self.population))])])

        # Perform crossover and mutation
        for i in range(len(selected)):
            if random.random() < 0.09259259259259259:
                parent1 = selected[np.random.randint(len(selected))]
                parent2 = selected[np.random.randint(len(selected))]
                child = (parent1 + parent2) / 2
                selected[i] = child

        # Update the bounds
        self.bounds = np.array([np.min(selected, axis=0), np.max(selected, axis=0)])

        # Evaluate the new candidates
        f_candidates = [func(candidate) for candidate in selected]
        f_evals = min(f_candidates)

        # Update the best solution if necessary
        if f_evals < self.f_best:
            self.f_best = f_evals
            self.x_best = selected[np.argmin(f_candidates)]
            self.f_evals_best = f_evals

        return self.x_best

# Example usage:
def func(x):
    return x[0]**2 + x[1]**2

ess = ESS(budget=10, dim=2)
x_opt = ess(func)
print(x_opt)