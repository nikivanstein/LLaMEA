class Improved_EA_ADES_Refined(Improved_EA_ADES):
    def __init__(self, budget, dim):
        super().__init__(budget, dim)
        self.ortho_prob = 0.1  # Initial probability for orthogonal evolution

    def __call__(self, func):
        for _ in range(self.budget):
            new_pop = np.empty_like(pop)
            for i in range(self.pop_size):
                strategy = np.random.choice(6, p=self.strategy_probs)
                trial = mutate(pop[i], pop, strategy)
                trial = np.clip(trial, -5.0, 5.0)
                trial = crossover(pop[i], trial)
                trial_fit = func(trial)
                if trial_fit < fitness[i]:
                    new_pop[i] = trial
                    fitness[i] = trial_fit
                    if strategy != 1:  # Adjust strategy probabilities
                        self.strategy_probs[strategy] += 0.1
                        self.strategy_probs[self.strategy_probs < 1e-6] = 1e-6
                        self.strategy_probs /= np.sum(self.strategy_probs)
                    if trial_fit < fitness[i]:  # Adaptive parameter control
                        self.mut_prob *= 1.2 if np.random.rand() < self.ortho_prob else 1.0
                        self.cross_prob *= 0.8 if np.random.rand() < self.ortho_prob else 1.0
                        self.ortho_prob = min(0.9, self.ortho_prob * 1.1)
                else:
                    new_pop[i] = pop[i]
            pop = new_pop