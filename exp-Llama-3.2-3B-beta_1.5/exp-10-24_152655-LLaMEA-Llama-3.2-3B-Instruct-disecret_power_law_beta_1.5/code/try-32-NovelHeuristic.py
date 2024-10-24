import numpy as np
import random
import operator

class NovelHeuristic:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.bounds = np.array([-5.0, 5.0]) * np.ones((dim, 1))
        self.f_best = np.inf
        self.x_best = np.zeros((dim, 1))
        self.f_evals = 0
        self.f_evals_best = 0
        self.evolution_strategies = [np.random.uniform(self.bounds[:, 0], self.bounds[:, 1], size=(self.dim, 1))]
        self.schedules = [np.random.uniform(self.bounds[:, 0], self.bounds[:, 1], size=(self.dim, 1))]

    def __call__(self, func):
        if self.f_evals >= self.budget:
            return self.x_best

        for _ in range(self.budget - self.f_evals):
            # Select evolutionary strategy and schedule
            strategy = random.choice(self.evolution_strategies)
            schedule = np.random.uniform(self.bounds[:, 0], self.bounds[:, 1], size=(self.dim, 1))

            # Evaluate the candidates
            f_candidates = func(strategy + schedule)

            # Update the best solution
            f_evals = f_candidates[0]
            x_best = strategy + schedule
            f_evals_best = f_evals

            # Update the best solution if necessary
            if f_evals < self.f_best:
                self.f_best = f_evals
                self.x_best = x_best
                self.f_evals_best = f_evals

            # Update evolutionary strategy and schedule
            self.evolution_strategies = [strategy, schedule]
            self.schedules = [strategy, schedule]

        return self.x_best

# Example usage:
def func(x):
    return x[0]**2 + x[1]**2

novel_heuristic = NovelHeuristic(budget=10, dim=2)
x_opt = novel_heuristic(func)
print(x_opt)