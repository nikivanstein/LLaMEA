import numpy as np

class EA_ADES:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 10 * dim
        self.cr = 0.5
        self.f = 0.5
        self.strategy_probs = np.full(6, 1/6)  # Initialize equal probabilities for 6 DE strategies

    def __call__(self, func):
        def mutate(x, pop, strategy, mut_prob):
            idxs = np.random.choice(len(pop), 3, replace=False)
            a, b, c = pop[idxs]
            return np.where(np.random.rand(self.dim) < mut_prob, x + self.f * (b - c), x)

        def crossover(x, trial, strategy, cross_prob):
            jrand = np.random.randint(self.dim)
            return np.where((np.random.rand(self.dim) < cross_prob) | (np.arange(self.dim) == jrand), trial, x)

        pop = np.random.uniform(-5.0, 5.0, (self.pop_size, self.dim))
        fitness = np.array([func(p) for p in pop])
        
        mut_prob = 0.5  # Initial mutation probability
        cross_prob = 0.9  # Initial crossover probability

        for _ in range(self.budget):
            new_pop = np.empty_like(pop)
            for i in range(self.pop_size):
                strategy = np.random.choice(6, p=self.strategy_probs)
                trial = mutate(pop[i], pop, strategy, mut_prob)
                trial = np.clip(trial, -5.0, 5.0)
                trial = crossover(pop[i], trial, strategy, cross_prob)
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