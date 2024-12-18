class EA_ADES:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 10 * dim
        self.adaptive_factor = 0.2  # Dynamic adjustment factor for mutation and crossover
        self.strategy_probs = np.full(6, 1/6)  # Initialize equal probabilities for 6 DE strategies

    def __call__(self, func):
        def mutate(x, pop, strategy):
            idxs = np.random.choice(len(pop), 3, replace=False)
            a, b, c = pop[idxs]
            if strategy == 0:  # DE/rand/1
                return a + self.adaptive_factor * (b - c)
            elif strategy == 1:  # DE/best/1
                return x + self.adaptive_factor * (pop[np.argmin([func(p) for p in pop])] - x)

        def crossover(x, trial, strategy):
            jrand = np.random.randint(self.dim)
            for j in range(self.dim):
                if np.random.rand() < self.adaptive_factor or j == jrand:
                    x[j] = trial[j]
            return x

        pop = np.random.uniform(-5.0, 5.0, (self.pop_size, self.dim))
        fitness = np.array([func(p) for p in pop])

        for _ in range(self.budget):
            new_pop = np.empty_like(pop)
            for i in range(self.pop_size):
                strategy = np.random.choice(6, p=self.strategy_probs)
                trial = mutate(pop[i], pop, strategy)
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