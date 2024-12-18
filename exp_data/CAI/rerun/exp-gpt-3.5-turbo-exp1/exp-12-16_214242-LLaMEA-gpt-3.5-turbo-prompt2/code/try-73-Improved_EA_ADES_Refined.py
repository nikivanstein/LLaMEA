import numpy as np

class Improved_EA_ADES_Refined(Improved_EA_ADES):
    def __init__(self, budget, dim):
        super().__init__(budget, dim)
        self.mut_prob_ind = np.full(self.pop_size, self.mut_prob)
        self.cross_prob_ind = np.full(self.pop_size, self.cross_prob)

    def __call__(self, func):
        def mutate(x, pop, strategy, mut_prob):
            idxs = np.random.choice(len(pop), 3, replace=False)
            a, b, c = pop[idxs]
            return x + self.f * (b - c) * mut_prob

        def crossover(x, trial, cross_prob):
            jrand = np.random.randint(self.dim)
            return np.where(np.random.uniform(0, 1, self.dim) < cross_prob, trial, x)

        for _ in range(self.budget):
            new_pop = np.empty_like(pop)
            for i in range(self.pop_size):
                strategy = np.random.choice(6, p=self.strategy_probs)
                trial = mutate(pop[i], pop, strategy, self.mut_prob)
                trial = np.clip(trial, -5.0, 5.0)
                trial = crossover(pop[i], trial, self.cross_prob)
                trial_fit = func(trial)
                if trial_fit < fitness[i]:
                    new_pop[i] = trial
                    fitness[i] = trial_fit
                    if strategy != 1:  # Adjust strategy probabilities
                        self.strategy_probs[strategy] += 0.1
                        self.strategy_probs[self.strategy_probs < 1e-6] = 1e-6
                        self.strategy_probs /= np.sum(self.strategy_probs)
                    if trial_fit < fitness[i]:  # Adaptive parameter control
                        self.mut_prob_ind[i] = min(1.0, self.mut_prob_ind[i] * 1.2)
                        self.cross_prob_ind[i] = max(0.1, self.cross_prob_ind[i] * 0.8)
                else:
                    new_pop[i] = pop[i]
            pop = new_pop

        best_idx = np.argmin(fitness)
        return pop[best_idx]