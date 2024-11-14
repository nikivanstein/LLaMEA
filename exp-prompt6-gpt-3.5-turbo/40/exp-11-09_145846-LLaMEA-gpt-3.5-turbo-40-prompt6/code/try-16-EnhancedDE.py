import numpy as np

class EnhancedDE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        population = np.random.uniform(-5.0, 5.0, (self.budget, self.dim))
        fitness = np.array([func(x) for x in population])
        best_index = np.argmin(fitness)
        best_solution = population[best_index]
        
        scaling_factor = 0.5
        crossover_rate = 0.9

        for _ in range(self.budget):
            diversity = np.mean(np.std(population, axis=0))
            scaling_factor = 0.5 + 0.2 * np.tanh(0.01 * diversity)
            crossover_rate = 0.9 - 0.4 * np.tanh(0.005 * diversity)

            for i in range(self.budget):
                a, b, c = np.random.choice(self.budget, 3, replace=False)
                mutant = population[a] + scaling_factor * (population[best_index] - population[a]) + scaling_factor * (population[b] - population[c])
                crossover = np.random.rand(self.dim) < crossover_rate
                trial = np.where(crossover, mutant, population[i])
                f_trial = func(trial)
                if f_trial < fitness[i]:
                    population[i] = trial
                    fitness[i] = f_trial
                    if f_trial < fitness[best_index]:
                        best_index = i
                        best_solution = trial

        return best_solution