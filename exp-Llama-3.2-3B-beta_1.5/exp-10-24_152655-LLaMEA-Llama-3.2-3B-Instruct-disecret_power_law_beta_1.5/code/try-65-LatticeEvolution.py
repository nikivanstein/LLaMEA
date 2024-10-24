import numpy as np
import random

class LatticeEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.bounds = np.array([-5.0, 5.0]) * np.ones((dim, 1))
        self.f_best = np.inf
        self.x_best = np.zeros((dim, 1))
        self.f_evals = 0
        self.f_evals_best = 0
        self.lattice_size = 10

    def __call__(self, func):
        if self.f_evals >= self.budget:
            return self.x_best

        for _ in range(self.budget - self.f_evals):
            # Initialize the lattice
            lattice = np.random.uniform(self.bounds[:, 0], self.bounds[:, 1], size=(self.lattice_size, self.dim))

            # Evaluate the lattice
            f_lattice = func(lattice)

            # Update the best solution
            f_evals = f_lattice[0]
            x_best = lattice[0]
            f_evals_best = f_evals

            # Update the best solution if necessary
            if f_evals < self.f_best:
                self.f_best = f_evals
                self.x_best = x_best
                self.f_evals_best = f_evals

            # Evolve the lattice
            new_lattice = np.zeros((self.lattice_size, self.dim))
            for i in range(self.lattice_size):
                for j in range(self.dim):
                    # Select two parents
                    parent1 = np.random.choice(lattice, 1, replace=False)[0]
                    parent2 = np.random.choice(lattice, 1, replace=False)[0]

                    # Perform crossover and mutation
                    child = np.zeros((self.dim, 1))
                    for k in range(self.dim):
                        if random.random() < 0.018518518518518517:
                            child[k] = (parent1[k] + parent2[k]) / 2
                        else:
                            child[k] = parent1[k]

                    # Evaluate the child
                    f_child = func(child)

                    # Update the new lattice
                    new_lattice[i, j] = child

            # Update the lattice
            lattice = new_lattice

        return self.x_best

# Example usage:
def func(x):
    return x[0]**2 + x[1]**2

lattice_evolution = LatticeEvolution(budget=10, dim=2)
x_opt = lattice_evolution(func)
print(x_opt)