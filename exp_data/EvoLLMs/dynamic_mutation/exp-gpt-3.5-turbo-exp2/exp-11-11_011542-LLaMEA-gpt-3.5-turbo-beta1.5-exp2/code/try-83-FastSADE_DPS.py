import numpy as np

class FastSADE_DPS(SADE_DPS):
    def __init__(self, budget, dim):
        super().__init__(budget, dim)
        self.adaptive_f = 0.5  # initial adaptation factor

    def __call__(self, func):
        def adaptive_mutation(population, best, scale_factor, crossover_rate):
            mutant_pop = []
            for idx, agent in enumerate(population):
                a, b, c = np.random.choice(population, 3, replace=False)
                adaptive_factor = max(min(1.0, self.adaptive_f + np.random.normal(0, 0.1)), 0.1)  # adaptive factor
                mutant = np.clip(a + adaptive_factor * scale_factor * (b - c), -5.0, 5.0)
                crossover_mask = np.random.rand(self.dim) < crossover_rate
                trial = np.where(crossover_mask, mutant, agent)
                if func(trial) < func(agent) and func(trial) < func(best):
                    best = trial
                mutant_pop.append(trial)
            return mutant_pop, best

        population = np.random.uniform(-5.0, 5.0, (self.pop_size, self.dim))
        best_agent = population[np.argmin([func(agent) for agent in population])]
        
        for _ in range(self.budget - self.pop_size):
            mutated_pop, best_agent = adaptive_mutation(population, best_agent, self.f, self.cr)
            population = mutated_pop

            if np.random.rand() < 0.1:  # adjust population size
                self.pop_size = min(max(int(self.pop_size * 1.1), 5), 50)
                population = np.vstack((population, np.random.uniform(-5.0, 5.0, (self.pop_size - len(population), self.dim))))

        self.adaptive_f *= 1.02  # adapt adaptation factor

        return best_agent