import numpy as np

class EnhancedBCDE(BCDE):
    def __init__(self, budget, dim, f=0.8, cr=0.9):
        super().__init__(budget, dim)
        self.f = f
        self.cr = cr

    def adaptive_mutation(self, best, x_r1, x_r2, f_min=0.2, f_max=0.9, cr_min=0.2, cr_max=0.9):
        f = f_min + (f_max - f_min) * np.random.rand()
        cr = cr_min + (cr_max - cr_min) * np.random.rand()
        return self.boundary_handling(best + f * (x_r1 - x_r2)), cr

    def __call__(self, func):
        population = np.random.uniform(self.lb, self.ub, (self.budget, self.dim))
        fitness = np.array([func(ind) for ind in population])

        for _ in range(self.budget):
            idx = np.argsort(fitness)
            best = population[idx[0]]

            mutant, cr = self.adaptive_mutation(best, population[idx[1]], population[idx[2])
            trial = np.where(np.random.uniform(0, 1, self.dim) < cr, mutant, population[idx[0]])

            trial_fitness = func(trial)
            if trial_fitness < fitness[idx[0]]:
                population[idx[0]] = trial
                fitness[idx[0]] = trial_fitness

        return population[idx[0]]