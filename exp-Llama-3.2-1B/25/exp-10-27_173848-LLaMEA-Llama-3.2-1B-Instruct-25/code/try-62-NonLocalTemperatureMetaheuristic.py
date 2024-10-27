import numpy as np
import random
import copy

class NonLocalTemperatureMetaheuristic:
    def __init__(self, budget, dim, alpha=0.5, mu=0.1, tau=0.9):
        self.budget = budget
        self.dim = dim
        self.alpha = alpha
        self.mu = mu
        self.tau = tau
        self.temp = 1.0
        self.best_func = None
        self.population = []

    def __call__(self, func):
        if self.budget <= 0:
            raise ValueError("Budget cannot be zero or negative")

        num_evals = 0
        while num_evals < self.budget and self.best_func is None:
            # Generate a random perturbation
            perturbation = np.random.uniform(-self.dim, self.dim)

            # Evaluate the new function
            new_func = copy.deepcopy(func)
            new_func += perturbation

            # Check if the new function is better
            if np.random.rand() < self.alpha:
                self.best_func = new_func
            else:
                # If the new function is not better, revert the perturbation
                perturbation *= self.tau
                new_func = copy.deepcopy(func)
                new_func -= perturbation

            num_evals += 1

        return self.best_func

    def mutate(self, individual):
        # Apply mutation strategy
        if random.random() < self.alpha:
            # Randomly select a mutation point
            mutation_point = random.randint(0, self.dim - 1)

            # Swap the elements at the mutation point
            individual[mutation_point], individual[mutation_point + 1] = individual[mutation_point + 1], individual[mutation_point]

            # Check if the mutation point is within the bounds
            if random.random() < self.mu:
                # If the mutation point is within the bounds, swap the elements again
                individual[mutation_point], individual[mutation_point + 1] = individual[mutation_point + 1], individual[mutation_point]

            # Check if the mutation point is within the bounds of the problem
            if random.random() < self.tau:
                # If the mutation point is within the bounds of the problem, swap the elements again
                individual[mutation_point], individual[mutation_point + 1] = individual[mutation_point + 1], individual[mutation_point]

            return individual
        else:
            return individual

    def __repr__(self):
        return f"NonLocalTemperatureMetaheuristic(budget={self.budget}, dim={self.dim}, alpha={self.alpha}, mu={self.mu}, tau={self.tau})"