import numpy as np
from scipy.optimize import differential_evolution

class DPHE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.adaptive_prob = 0.45

    def __call__(self, func):
        def neg_func(x):
            return -func(x)

        bounds = [(self.lower_bound, self.upper_bound) for _ in range(self.dim)]
        res = differential_evolution(func, bounds, x0=np.random.uniform(self.lower_bound, self.upper_bound, size=self.dim), maxiter=self.budget, tol=1e-6)

        if res.success:
            # Select the best individual and refine its strategy
            best_individual = res.x
            # Generate a new individual with an adaptive mutation probability
            new_individual = self.refine_strategy(best_individual, self.adaptive_prob)
            return new_individual
        else:
            return None

    def refine_strategy(self, individual, prob):
        # Calculate the mutation probability for each dimension
        mutation_prob = np.random.uniform(0, prob, size=self.dim)
        # Select the dimensions with the highest mutation probability
        selected_dims = np.where(mutation_prob > prob / 2)[0]
        # Refine the strategy by perturbing the selected dimensions
        new_individual = individual.copy()
        for dim in selected_dims:
            # Perturb the dimension with a probability of 1 - prob
            if np.random.rand() < 1 - prob:
                new_individual[dim] += np.random.uniform(-1, 1)
        return new_individual