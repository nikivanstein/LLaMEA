class EnhancedDynamicDE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        # Initialize population
        population = np.random.uniform(-5.0, 5.0, (self.budget, self.dim))
        fitness = np.array([func(ind) for ind in population])
        
        for _ in range(self.budget):
            for i in range(self.budget):
                F = 0.5 + np.random.random() / 2.0 * (1.0 - np.tanh(np.mean(fitness)))
                diversity = np.std(population)
                mutant = population[i] + F * (population[i-1] - population[i-2]) + np.random.standard_cauchy(self.dim) * diversity
                trial = mutant if func(mutant) < fitness[i] else population[i]

                if func(trial) < fitness[i]:
                    population[i] = trial
                    fitness[i] = func(trial)

        return population[np.argmin(fitness)]