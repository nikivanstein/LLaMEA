import numpy as np

class EnhancedAdaptiveDE(ImprovedAdaptiveDE):
    def __init__(self, budget, dim):
        super().__init__(budget, dim)
        self.F_min, self.F_max = 0.2, 0.8
        self.CR_min, self.CR_max = 0.2, 0.8
        self.strategy_weights = np.array([0.3, 0.3, 0.4])  # Define weights for mutation strategies

    def mutate_individual(self, pop, i, F):
        strategy = np.random.choice(['best', 'current', 'rand'], p=self.strategy_weights)
        if strategy == 'best':
            return pop[np.argmax(self.evaluate_pop(pop))]
        elif strategy == 'current':
            return pop[i]
        else:
            return pop[np.random.choice(len(pop))]

    def crossover_individual(self, target, mutant, CR):
        trial = target.copy()
        mask = np.random.rand(len(target)) < CR
        trial[mask] = mutant[mask]
        return trial

    def dynamic_population_size(self, t):
        return int(10 + 40 * (1 - np.exp(-t / 800)))

    def __call__(self, func):
        return super().__call__(func)