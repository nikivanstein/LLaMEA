import numpy as np
import random

class HyperEdge:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.bounds = np.array([-5.0, 5.0]) * np.ones((dim, 1))
        self.f_best = np.inf
        self.x_best = np.zeros((dim, 1))
        self.f_evals = 0
        self.f_evals_best = 0
        self.hyper_edges = []

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

            # Construct the hyper-edge structure
            for i in range(self.dim):
                for j in range(self.dim):
                    if i!= j:
                        # Create a new hyper-edge
                        new_edge = {
                            'i': i,
                            'j': j,
                            'f_min': np.inf,
                            'f_max': -np.inf,
                            'x_best': None
                        }

                        # Find the best point on the hyper-edge
                        for x in np.linspace(self.bounds[i, 0], self.bounds[i, 1], 100):
                            for y in np.linspace(self.bounds[j, 0], self.bounds[j, 1], 100):
                                f = func(np.array([[x], [y]]))
                                if f < new_edge['f_min']:
                                    new_edge['f_min'] = f
                                    new_edge['x_best'] = np.array([[x], [y]])
                                if f > new_edge['f_max']:
                                    new_edge['f_max'] = f

                        # Add the hyper-edge to the list
                        self.hyper_edges.append(new_edge)

            # Select the best hyper-edge
            best_hyper_edge = self.hyper_edges[np.argmin([edge['f_min'] for edge in self.hyper_edges])]

            # Schedule the best hyper-edge
            for edge in self.hyper_edges:
                if edge['i'] == best_hyper_edge['i'] and edge['j'] == best_hyper_edge['j']:
                    candidates = np.delete(candidates, np.where(candidates == edge['x_best']), axis=0)

        return self.x_best

# Example usage:
def func(x):
    return x[0]**2 + x[1]**2

hyper_edge = HyperEdge(budget=10, dim=2)
x_opt = hyper_edge(func)
print(x_opt)