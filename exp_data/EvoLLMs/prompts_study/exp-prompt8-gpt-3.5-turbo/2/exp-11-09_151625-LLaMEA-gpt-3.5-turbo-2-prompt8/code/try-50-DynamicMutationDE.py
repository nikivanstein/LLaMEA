class DynamicMutationDE(ProbabilisticMutationDE):
    def __init__(self, budget, dim):
        super().__init__(budget, dim)
        self.F_history = np.full(self.budget, self.F)

    def __call__(self, func):
        population = np.random.uniform(-5.0, 5.0, (self.budget, self.dim))
        fitness = np.array([func(x) for x in population])
        
        for i in range(self.budget):
            target = population[i]
            idxs = np.arange(self.budget)
            np.random.shuffle(idxs)
            a, b, c = population[np.random.choice(idxs[:3], 3, replace=False)]
            mutant = a + self.F_history[i] * (b - c)
            crossover = np.random.rand(self.dim) < self.CR
            trial = np.where(crossover, mutant, target)
            
            trial_fitness = func(trial)
            if trial_fitness < fitness[i]:
                population[i] = trial
                fitness[i] = trial_fitness
                if np.random.rand() < 0.1:
                    self.F_history[i] = np.clip(np.random.normal(self.F_history[i], 0.1), 0.1, 0.9)
                    
        best_idx = np.argmin(fitness)
        return population[best_idx]