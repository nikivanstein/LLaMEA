import numpy as np

class Hybrid_DE_PSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 10  # initial population size
        self.c1 = 1.5
        self.c2 = 1.5
        self.w = 0.7
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

        population = np.random.uniform(-5.0, 5.0, (self.pop_size, self.dim))
        best_agent = population[np.argmin([func(agent) for agent in population])]
        
        for _ in range(self.budget - self.pop_size):
            mutated_pop, best_agent = mutation(population, best_agent, self.f, self.cr)
            population = mutated_pop

            pbest = population[np.argmin([func(agent) for agent in population])]
            for idx, agent in enumerate(population):
                r1, r2 = np.random.uniform(0, 1, self.dim), np.random.uniform(0, 1, self.dim)
                velocity = self.w * velocity + self.c1 * r1 * (pbest - agent) + self.c2 * r2 * (best_agent - agent)
                agent = agent + velocity
                agent = np.clip(agent, -5.0, 5.0)
                population[idx] = agent

            if np.random.rand() < 0.1:  # adjust population size
                self.pop_size = min(max(int(self.pop_size * 1.1), 5), 50)
                population = np.vstack((population, np.random.uniform(-5.0, 5.0, (self.pop_size - len(population), self.dim))))

        return best_agent
algorithm(problem)