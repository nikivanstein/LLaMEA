import numpy as np

class Dynamic_SADE_DPS(SADE_DPS):
    def __init__(self, budget, dim):
        super().__init__(budget, dim)
        self.mutation_strategy = np.ones(self.pop_size)

    def __call__(self, func):
        def dynamic_mutation(population, best, scale_factor, crossover_rate, mutation_strategy):
            mutant_pop = []
            for idx, agent in enumerate(population):
                a, b, c = np.random.choice(population, 3, replace=False)
                mutant = np.clip(a + mutation_strategy[idx] * (b - c), -5.0, 5.0)
                crossover_mask = np.random.rand(self.dim) < crossover_rate
                trial = np.where(crossover_mask, mutant, agent)
                if func(trial) < func(agent) and func(trial) < func(best):
                    best = trial
                    mutation_strategy[idx] *= 1.1  # Adjust mutation strategy based on performance
                else:
                    mutation_strategy[idx] *= 0.9
                mutant_pop.append(trial)
            return mutant_pop, best, mutation_strategy

        population = np.random.uniform(-5.0, 5.0, (self.pop_size, self.dim))
        best_agent = population[np.argmin([func(agent) for agent in population])
        for _ in range(self.budget - self.pop_size):
            mutated_pop, best_agent, self.mutation_strategy = dynamic_mutation(population, best_agent, self.f, self.cr, self.mutation_strategy)
            population = mutated_pop

            if np.random.rand() < 0.1:  # Adjust population size
                self.pop_size = min(max(int(self.pop_size * 1.1), 5), 50)
                population = np.vstack((population, np.random.uniform(-5.0, 5.0, (self.pop_size - len(population), self.dim))))

        return best_agent