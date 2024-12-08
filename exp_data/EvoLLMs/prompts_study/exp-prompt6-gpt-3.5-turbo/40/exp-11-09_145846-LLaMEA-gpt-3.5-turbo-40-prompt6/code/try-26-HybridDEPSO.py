import numpy as np

class HybridDEPSO:
    def __init__(self, budget, dim, F=0.5, CR=0.9, w=0.5, c1=1.496, c2=1.496):
        self.budget = budget
        self.dim = dim
        self.F = F
        self.CR = CR
        self.w = w
        self.c1 = c1
        self.c2 = c2

    def __call__(self, func):
        population = np.random.uniform(-5.0, 5.0, (self.budget, self.dim))
        velocity = np.zeros((self.budget, self.dim))
        fitness = np.array([func(x) for x in population])
        best_index = np.argmin(fitness)
        best_solution = population[best_index]

        for _ in range(self.budget):
            for i in range(self.budget):
                a, b, c = np.random.choice(self.budget, 3, replace=False)
                mutant_de = population[a] + self.F * (population[best_index] - population[a]) + self.F * (population[b] - population[c])

                velocity[i] = self.w * velocity[i] + self.c1 * np.random.rand(self.dim) * (best_solution - population[i]) + self.c2 * np.random.rand(self.dim) * (mutant_de - population[i])
                trial = population[i] + velocity[i]

                f_trial = func(trial)
                if f_trial < fitness[i]:
                    population[i] = trial
                    fitness[i] = f_trial
                    if f_trial < fitness[best_index]:
                        best_index = i
                        best_solution = trial

        return best_solution