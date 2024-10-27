import numpy as np
from scipy.optimize import differential_evolution

class DPHE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.perturbation_prob = 0.45

    def __call__(self, func):
        def neg_func(x):
            return -func(x)

        bounds = [(self.lower_bound, self.upper_bound) for _ in range(self.dim)]
        res = differential_evolution(func, bounds, x0=np.random.uniform(self.lower_bound, self.upper_bound, size=self.dim), maxiter=self.budget, tol=1e-6)

        if res.success:
            # Refine the solution with probability 0.45
            if np.random.rand() < self.perturbation_prob:
                # Generate a new individual by perturbing the current solution
                new_individual = self.perturb_individual(res.x)
                # Evaluate the fitness of the new individual
                new_fitness = self.evaluate_fitness(new_individual, func)
                # Replace the current solution with the new individual if it has better fitness
                if new_fitness < func(res.x):
                    return new_individual
            return res.x
        else:
            return None

    def perturb_individual(self, individual):
        # Generate a new individual by perturbing the current solution
        new_individual = individual + np.random.uniform(-1.0, 1.0, size=self.dim)
        # Ensure the new individual is within the bounds
        new_individual = np.clip(new_individual, self.lower_bound, self.upper_bound)
        return new_individual

    def evaluate_fitness(self, individual, func):
        # Evaluate the fitness of the individual
        fitness = func(individual)
        return fitness