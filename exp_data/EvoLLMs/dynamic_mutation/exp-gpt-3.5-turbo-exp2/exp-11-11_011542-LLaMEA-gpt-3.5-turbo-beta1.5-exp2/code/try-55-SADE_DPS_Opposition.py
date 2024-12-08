import numpy as np

class SADE_DPS_Opposition:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 10  # initial population size
        self.cr = 0.9
        self.f = 0.5

    def __call__(self, func):
        def mutation(population, best, scale_factor, crossover_rate):
            mutant_pop = []
            for idx, agent in enumerate(population):
                a, b, c = np.random.choice(population, 3, replace=False)
                mutant = np.clip(a + scale_factor * (b - c), -5.0, 5.0)
                crossover_mask = np.random.rand(self.dim) < crossover_rate
                trial = np.where(crossover_mask, mutant, agent)
                if func(trial) < func(agent) and func(trial) < func(best):
                    best = trial
                mutant_pop.append(trial)
            return mutant_pop, best

        def opposition(population):
            return 2.0 * np.mean(population, axis=0) - population

        population = np.random.uniform(-5.0, 5.0, (self.pop_size, self.dim))
        best_agent = population[np.argmin([func(agent) for agent in population])
        
        for _ in range(self.budget - self.pop_size):
            mutated_pop, best_agent = mutation(population, best_agent, self.f, self.cr)
            population = np.vstack((population, np.random.uniform(-5.0, 5.0, (self.pop_size - len(population), self.dim))))

            opp_population = opposition(population)
            opp_mutated_pop, best_agent = mutation(opp_population, best_agent, self.f, self.cr)
            population = np.vstack((population, opp_mutated_pop))

        return best_agent