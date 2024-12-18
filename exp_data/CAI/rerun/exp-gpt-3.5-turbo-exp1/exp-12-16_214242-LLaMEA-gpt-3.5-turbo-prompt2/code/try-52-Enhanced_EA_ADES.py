class Enhanced_EA_ADES(Improved_EA_ADES):
    def __init__(self, budget, dim):
        super().__init__(budget, dim)

    def __call__(self, func):
        pop = np.random.uniform(-5.0, 5.0, (self.pop_size, self.dim))
        fitness = np.array([func(p) for p in pop])

        for _ in range(self.budget):
            new_pop = np.empty_like(pop)
            diversity_scores = [np.mean(np.linalg.norm(pop - pop[i], axis=1)) for i in range(self.pop_size)]
            diversity_scores /= np.sum(diversity_scores)  # Normalize diversity scores
            for i in range(self.pop_size):
                selected = np.random.choice(self.pop_size, p=diversity_scores)
                strategy = np.random.choice(6, p=self.strategy_probs)
                trial = mutate(pop[selected], pop, strategy)
                trial = np.clip(trial, -5.0, 5.0)
                trial = crossover(pop[selected], trial)
                trial_fit = func(trial)
                if trial_fit < fitness[selected]:
                    new_pop[selected] = trial
                    fitness[selected] = trial_fit
                    if strategy != 1:
                        self.strategy_probs[strategy] += 0.1
                        self.strategy_probs[self.strategy_probs < 1e-6] = 1e-6
                        self.strategy_probs /= np.sum(self.strategy_probs)
                    if trial_fit < fitness[selected]:
                        self.mut_prob = min(1.0, self.mut_prob * 1.2)
                        self.cross_prob = max(0.1, self.cross_prob * 0.8)
                else:
                    new_pop[selected] = pop[selected]
            pop = new_pop

        best_idx = np.argmin(fitness)
        return pop[best_idx]