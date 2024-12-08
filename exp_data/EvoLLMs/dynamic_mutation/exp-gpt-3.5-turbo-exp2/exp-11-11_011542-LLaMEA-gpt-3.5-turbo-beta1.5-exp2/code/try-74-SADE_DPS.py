class SADE_DPS:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 10
        self.cr = 0.9
        self.f = 0.5
        self.mut_prob = 0.5  # initial mutation probability

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
        best_agent = population[np.argmin([func(agent) for agent in population])
        performance_history = [float('inf')] * self.pop_size

        for _ in range(self.budget - self.pop_size):
            mutated_pop, best_agent = mutation(population, best_agent, self.f, self.cr)
            population = mutated_pop
            perf_improve_count = sum(func(population[i]) < performance_history[i] for i in range(self.pop_size))
            self.mut_prob *= 1.1 if perf_improve_count > self.pop_size * 0.8 else 0.9
            self.mut_prob = max(min(self.mut_prob, 1.0), 0.1)

            if np.random.rand() < self.mut_prob:
                self.f = np.clip(self.f + np.random.normal(0, 0.1), 0.1, 0.9)
                self.cr = np.clip(self.cr + np.random.normal(0, 0.1), 0.1, 0.9)

        return best_agent