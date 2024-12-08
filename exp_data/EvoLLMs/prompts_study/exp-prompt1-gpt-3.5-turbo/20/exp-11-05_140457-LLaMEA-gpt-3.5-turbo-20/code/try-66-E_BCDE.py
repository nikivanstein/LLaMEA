import numpy as np

class E_BCDE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lb = -5.0
        self.ub = 5.0

    def boundary_handling(self, x):
        return np.clip(x, self.lb, self.ub)

    def __call__(self, func):
        population = np.random.uniform(self.lb, self.ub, (self.budget, self.dim))
        fitness = np.array([func(ind) for ind in population])

        for _ in range(self.budget):
            idx = np.argsort(fitness)
            best = population[idx[0]]

            mutation_prob = 0.8 + 0.2*np.exp(-_/(0.2*self.budget))  # Adaptive mutation probability
            mutant = self.boundary_handling(best + mutation_prob * (population[idx[1]] - population[idx[2]])
            trial = np.where(np.random.uniform(0, 1, self.dim) < mutation_prob, mutant, population[idx[0]])

            trial_fitness = func(trial)
            if trial_fitness < fitness[idx[0]]:
                population[idx[0]] = trial
                fitness[idx[0]] = trial_fitness

        return population[idx[0]]