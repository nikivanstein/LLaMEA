class Adaptive_DEA(Improved_EA_ADES):
    def __init__(self, budget, dim):
        super().__init__(budget, dim)
        self.mut_factors = np.full(self.pop_size, 0.5)

    def __call__(self, func):
        def mutate(x, pop, strategy, i):
            idxs = np.random.choice(len(pop), 3, replace=False)
            a, b, c = pop[idxs]
            return x + self.mut_factors[i] * (b - c)

        pop = np.random.uniform(-5.0, 5.0, (self.pop_size, self.dim))

        for _ in range(self.budget):
            new_pop = np.empty_like(pop)
            for i in range(self.pop_size):
                strategy = np.random.choice(6, p=self.strategy_probs)
                trial = mutate(pop[i], pop, strategy, i)
                trial = np.clip(trial, -5.0, 5.0)
                trial = crossover(pop[i], trial)
                trial_fit = func(trial)
                if trial_fit < fitness[i]:
                    new_pop[i] = trial
                    fitness[i] = trial_fit
                    if strategy != 1:
                        self.strategy_probs[strategy] += 0.1
                        self.strategy_probs[self.strategy_probs < 1e-6] = 1e-6
                        self.strategy_probs /= np.sum(self.strategy_probs)
                    if trial_fit < fitness[i]:
                        self.mut_factors[i] = min(1.0, self.mut_factors[i] * 1.2)
                        self.cross_prob = max(0.1, self.cross_prob * 0.8)
                else:
                    new_pop[i] = pop[i]
            pop = new_pop

        best_idx = np.argmin(fitness)
        return pop[best_idx]