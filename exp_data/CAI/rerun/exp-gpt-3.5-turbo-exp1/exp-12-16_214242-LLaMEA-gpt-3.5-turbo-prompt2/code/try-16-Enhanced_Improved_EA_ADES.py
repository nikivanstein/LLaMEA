class Enhanced_Improved_EA_ADES(Improved_EA_ADES):
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
                    self.strategy_probs[strategy] = max(1e-6, self.strategy_probs[strategy] + 0.1)  # Update strategy probabilities
                    if trial_fit < fitness[i]:  # Adaptive parameter control
                        self.mut_prob = min(1.0, self.mut_prob * 1.2)
                        self.cross_prob = max(0.1, self.cross_prob * 0.8)
                else:
                    new_pop[i] = pop[i]
            pop = new_pop
        best_idx = np.argmin(fitness)
        return pop[best_idx]