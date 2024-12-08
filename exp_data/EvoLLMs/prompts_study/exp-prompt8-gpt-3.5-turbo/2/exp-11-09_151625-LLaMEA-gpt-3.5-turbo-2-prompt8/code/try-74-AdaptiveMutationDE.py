class AdaptiveMutationDE(ProbabilisticMutationDE):
    def __init__(self, budget, dim):
        super().__init__(budget, dim)
        self.weights = np.ones(self.budget)
    
    def __call__(self, func):
        population = np.random.uniform(-5.0, 5.0, (self.budget, self.dim))
        fitness = np.array([func(x) for x in population])
        
        for i in range(self.budget):
            target = population[i]
            idxs = np.arange(self.budget)
            np.random.shuffle(idxs)
            a, b, c = population[np.random.choice(idxs[:3], 3, replace=False)]
            mutant = a + self.F * (b - c)
            crossover = np.random.rand(self.dim) < self.CR
            trial = np.where(crossover, mutant, target)
            
            trial_fitness = func(trial)
            if trial_fitness < fitness[i]:
                population[i] = trial
                fitness[i] = trial_fitness
                self.weights[i] = min(1.0, self.weights[i] + 0.1) if trial_fitness < fitness[i] else max(0.1, self.weights[i] - 0.1)
                self.F = np.clip(np.random.normal(0.5, 0.1) * self.weights[i], 0.1, 0.9)
        
        best_idx = np.argmin(fitness)
        return population[best_idx]