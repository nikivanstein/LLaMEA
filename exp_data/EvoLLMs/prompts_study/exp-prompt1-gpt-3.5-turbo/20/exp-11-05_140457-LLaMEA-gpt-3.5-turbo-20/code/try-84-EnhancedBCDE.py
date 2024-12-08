import numpy as np

class EnhancedBCDE(BCDE):
    def __init__(self, budget, dim):
        super().__init__(budget, dim)
    
    def dynamic_mutation(self, diversity):
        return 0.8 + 0.2 * np.tanh(diversity)

    def __call__(self, func):
        population = np.random.uniform(self.lb, self.ub, (self.budget, self.dim))
        fitness = np.array([func(ind) for ind in population])
        diversity = np.std(population)

        for _ in range(self.budget):
            idx = np.argsort(fitness)
            best = population[idx[0]]

            scaling_factor = self.dynamic_mutation(diversity)
            mutant = self.boundary_handling(best + scaling_factor * (population[idx[1]] - population[idx[2]]))
            trial = np.where(np.random.uniform(0, 1, self.dim) < 0.9, mutant, population[idx[0]])

            trial_fitness = func(trial)
            if trial_fitness < fitness[idx[0]]:
                population[idx[0]] = trial
                fitness[idx[0]] = trial_fitness

            diversity = np.std(population)

        return population[idx[0]]