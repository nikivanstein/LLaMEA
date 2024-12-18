class Improved_EA_ADES_Enhanced(Improved_EA_ADES):
    def __call__(self, func):
        def dynamic_mutate(x, global_best):
            idxs = np.random.choice(len(pop), 3, replace=False)
            a, b, c = pop[idxs]
            return x + self.f * (b - c) + np.random.uniform(0, 1) * (global_best - x)

        pop = np.random.uniform(-5.0, 5.0, (self.pop_size, self.dim))
        fitness = np.array([func(p) for p in pop])
        global_best = pop[np.argmin(fitness)]

        for _ in range(self.budget):
            new_pop = np.empty_like(pop)
            for i in range(self.pop_size):
                strategy = np.random.choice(6, p=self.strategy_probs)
                trial = dynamic_mutate(pop[i], global_best)
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
                        self.mut_prob = min(1.0, self.mut_prob * 1.2)
                        self.cross_prob = max(0.1, self.cross_prob * 0.8)
                else:
                    new_pop[i] = pop[i]
            pop = new_pop
            global_best = pop[np.argmin(fitness)]

        best_idx = np.argmin(fitness)
        return pop[best_idx]