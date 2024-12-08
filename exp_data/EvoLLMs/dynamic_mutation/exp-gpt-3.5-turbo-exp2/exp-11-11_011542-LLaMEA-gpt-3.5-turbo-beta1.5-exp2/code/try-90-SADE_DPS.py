class SADE_DPS:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 10  # initial population size
        self.cr = 0.9
        self.f = 0.5

    def __call__(self, func):
        def mutation(population, best, scale_factor, crossover_rate, fitness_values):
            mutant_pop = []
            for idx, agent in enumerate(population):
                a, b, c = np.random.choice(population, 3, replace=False)
                if fitness_values[idx] > np.mean(fitness_values):  # Dynamic mutation rate based on fitness
                    scale_factor = np.clip(scale_factor * 1.2, 0.1, 2.0)  # Adjust scale factor
                mutant = np.clip(a + scale_factor * (b - c), -5.0, 5.0)
                crossover_mask = np.random.rand(self.dim) < crossover_rate
                trial = np.where(crossover_mask, mutant, agent)
                if func(trial) < func(agent) and func(trial) < func(best):
                    best = trial
                mutant_pop.append(trial)
            return mutant_pop, best

        population = np.random.uniform(-5.0, 5.0, (self.pop_size, self.dim))
        best_agent = population[np.argmin([func(agent) for agent in population])]
        fitness_values = [func(agent) for agent in population]

        for _ in range(self.budget - self.pop_size):
            mutated_pop, best_agent = mutation(population, best_agent, self.f, self.cr, fitness_values)
            population = mutated_pop

            if np.random.rand() < 0.1:  # adjust population size
                self.pop_size = min(max(int(self.pop_size * 1.1), 5), 50)
                population = np.vstack((population, np.random.uniform(-5.0, 5.0, (self.pop_size - len(population), self.dim))))
                fitness_values.extend([func(agent) for agent in population[len(fitness_values):]])

        return best_agent