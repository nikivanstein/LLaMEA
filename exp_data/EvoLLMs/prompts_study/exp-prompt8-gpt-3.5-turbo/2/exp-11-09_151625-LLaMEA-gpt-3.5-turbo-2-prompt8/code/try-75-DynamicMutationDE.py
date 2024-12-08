class DynamicMutationDE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.CR = 0.9
        self.F_min = 0.1
        self.F_max = 0.9

    def __call__(self, func):
        population = np.random.uniform(-5.0, 5.0, (self.budget, self.dim))
        fitness = np.array([func(x) for x in population])
        
        for i in range(self.budget):
            target = population[i]
            idxs = np.arange(self.budget)
            np.random.shuffle(idxs)
            a, b, c = population[np.random.choice(idxs[:3], 3, replace=False)]
            F = self.F_min + (self.F_max - self.F_min) * (i / self.budget)  # Dynamic mutation factor
            mutant = a + F * (b - c)
            crossover = np.random.rand(self.dim) < self.CR
            trial = np.where(crossover, mutant, target)
            
            trial_fitness = func(trial)
            if trial_fitness < fitness[i]:
                population[i] = trial
                fitness[i] = trial_fitness
        best_idx = np.argmin(fitness)
        return population[best_idx]