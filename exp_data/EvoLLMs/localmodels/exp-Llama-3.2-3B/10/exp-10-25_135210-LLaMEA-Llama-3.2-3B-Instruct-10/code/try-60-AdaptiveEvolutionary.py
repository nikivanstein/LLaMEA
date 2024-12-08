import numpy as np
import scipy.optimize as optimize

class AdaptiveEvolutionary:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.bounds = [(-5.0, 5.0)] * dim
        self.x_best = np.random.uniform(self.bounds[0][0], self.bounds[0][1], dim)
        self.f_best = np.inf
        self.x_grad = np.zeros((dim, dim))
        self.x_hessian = np.zeros((dim, dim, dim, dim))
        self.population = []
        self.selection_rate = 0.1

    def __call__(self, func):
        for _ in range(self.budget):
            # Select parents
            parents = np.random.choice(len(self.population), int(self.budget * 0.1), replace=False)
            selected_parents = [self.population[parent] for parent in parents]

            # Perform crossover
            offspring = []
            for _ in range(int(self.budget * 0.9)):
                parent1, parent2 = np.random.choice(selected_parents, 2, replace=False)
                child = np.random.uniform(self.bounds[0][0], self.bounds[0][1], self.dim)
                for i in range(self.dim):
                    if np.random.rand() < 0.5:
                        child[i] = (parent1[i] + parent2[i]) / 2
                offspring.append(child)

            # Perform mutation
            mutated_offspring = []
            for child in offspring:
                if np.random.rand() < self.selection_rate:
                    mutated_child = child + np.random.uniform(-1, 1, self.dim)
                    mutated_child = np.clip(mutated_child, self.bounds[0][0], self.bounds[0][1])
                    mutated_offspring.append(mutated_child)
                else:
                    mutated_offspring.append(child)

            # Evaluate offspring
            for child in mutated_offspring:
                f = func(child)
                if f < self.f_best:
                    self.x_best = child
                    self.f_best = f

                # Compute gradient
                x_grad = np.zeros(self.dim)
                for i in range(self.dim):
                    x_plus_epsilon = child.copy()
                    x_plus_epsilon[i] += 1e-6
                    f_plus_epsilon = func(x_plus_epsilon)
                    x_minus_epsilon = child.copy()
                    x_minus_epsilon[i] -= 1e-6
                    f_minus_epsilon = func(x_minus_epsilon)
                    x_grad[i] = (f_plus_epsilon - f_minus_epsilon) / (2 * 1e-6)

                # Compute Hessian
                x_hessian = np.zeros((self.dim, self.dim, self.dim, self.dim))
                for i in range(self.dim):
                    for j in range(self.dim):
                        for k in range(self.dim):
                            x_plus_epsilon = child.copy()
                            x_plus_epsilon[i] += 1e-6
                            x_plus_epsilon[j] += 1e-6
                            x_plus_epsilon[k] += 1e-6
                            f_plus_epsilon = func(x_plus_epsilon)
                            x_hessian[i, j, k, i] = (f_plus_epsilon - func(child)) / (6 * 1e-6**3)
                            x_plus_epsilon = child.copy()
                            x_plus_epsilon[i] -= 1e-6
                            x_plus_epsilon[j] += 1e-6
                            x_plus_epsilon[k] += 1e-6
                            f_plus_epsilon = func(x_plus_epsilon)
                            x_hessian[i, j, k, i] -= (f_plus_epsilon - func(child)) / (6 * 1e-6**3)
                            x_plus_epsilon = child.copy()
                            x_plus_epsilon[i] += 1e-6
                            x_plus_epsilon[j] -= 1e-6
                            x_plus_epsilon[k] += 1e-6
                            f_plus_epsilon = func(x_plus_epsilon)
                            x_hessian[i, j, k, i] += (f_plus_epsilon - func(child)) / (6 * 1e-6**3)
                            x_plus_epsilon = child.copy()
                            x_plus_epsilon[i] += 1e-6
                            x_plus_epsilon[j] += 1e-6
                            x_plus_epsilon[k] -= 1e-6
                            f_plus_epsilon = func(x_plus_epsilon)
                            x_hessian[i, j, k, i] -= (f_plus_epsilon - func(child)) / (6 * 1e-6**3)

            # Update population
            self.population = mutated_offspring

# Test the algorithm
def func(x):
    return np.sum(x**2)

budget = 100
dim = 10
alg = AdaptiveEvolutionary(budget, dim)
alg()