import numpy as np

class EnhancedFastConvergingDE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        population = np.random.uniform(-5.0, 5.0, (self.budget, self.dim))
        fitness = np.array([func(x) for x in population])
        best_index = np.argmin(fitness)
        best_solution = population[best_index]
        
        F = 0.5
        CR = 0.9

        for _ in range(self.budget):
            diversity = np.mean(np.std(population, axis=0))
            F = max(0.1, min(0.9, F + 0.1 * np.tanh((1 - diversity) * 10)))
            CR = max(0.1, min(0.9, CR + 0.1 * np.tanh((1 - diversity) * 10)))

            for i in range(self.budget):
                a, b, c = np.random.choice(self.budget, 3, replace=False)
                mutant = population[a] + F * (population[best_index] - population[a]) + F * (population[b] - population[c])
                crossover = np.random.rand(self.dim) < CR
                trial = np.where(crossover, mutant, population[i])
                f_trial = func(trial)
                if f_trial < fitness[i]:
                    population[i] = trial
                    fitness[i] = f_trial
                    if f_trial < fitness[best_index]:
                        best_index = i
                        best_solution = trial

        return best_solution