class Improved_EA_ADES(EA_ADES):
    def __init__(self, budget, dim):
        super().__init__(budget, dim)
        self.mutation_strategies = [self.mutate_rand_2, self.mutate_best_2]
        
    def mutate_rand_2(self, x, pop):
        idxs = np.random.choice(len(pop), 5, replace=False)
        a, b, c, d, e = pop[idxs]
        return a + self.f * (b - c) + self.f * (d - e)
    
    def mutate_best_2(self, x, pop):
        best_idx = np.argmin([func(p) for p in pop])
        best = pop[best_idx]
        idxs = np.random.choice(len(pop), 3, replace=False)
        a, b, c = pop[idxs]
        return x + self.f * (best - x) + self.f * (a - b) + self.f * (c - best)
    
    def __call__(self, func):
        pop = np.random.uniform(-5.0, 5.0, (self.pop_size, self.dim))
        fitness = np.array([func(p) for p in pop])
        
        for _ in range(self.budget):
            new_pop = np.empty_like(pop)
            for i in range(self.pop_size):
                strategy = np.random.choice(len(self.mutation_strategies))
                trial = self.mutation_strategies[strategy](pop[i], pop)
                trial = np.clip(trial, -5.0, 5.0)
                trial_fit = func(trial)
                if trial_fit < fitness[i]:
                    new_pop[i] = trial
                    fitness[i] = trial_fit
                    if strategy != 1:  # Adjust strategy probabilities
                        self.strategy_probs[strategy] += 0.1
                        self.strategy_probs[self.strategy_probs < 1e-6] = 1e-6
                        self.strategy_probs /= np.sum(self.strategy_probs)
                else:
                    new_pop[i] = pop[i]
            pop = new_pop

        best_idx = np.argmin(fitness)
        return pop[best_idx]