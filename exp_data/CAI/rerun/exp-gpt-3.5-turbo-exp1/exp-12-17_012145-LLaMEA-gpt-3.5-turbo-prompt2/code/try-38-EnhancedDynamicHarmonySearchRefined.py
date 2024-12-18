class EnhancedDynamicHarmonySearchRefined:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.harmony_memory_size = 10
        self.par_min = 0.1
        self.par_max = 0.9
        self.bandwidth_min = 0.01
        self.bandwidth_max = 0.1
        self.mutation_factor = 0.5
        self.crossover_prob = 0.7
        self.inertia_weight = 0.7
        self.c1 = 1.5
        self.c2 = 1.5
        self.mutation_factor_history = [self.mutation_factor]
        self.crossover_prob_history = [self.crossover_prob]

    def __call__(self, func):
        def update_parameters(iteration):
            par = self.par_min + (self.par_max - self.par_min) * (iteration / self.budget)
            bandwidth = self.bandwidth_min + (self.bandwidth_max - self.bandwidth_min) * (iteration / self.budget)
            return par, bandwidth

        def update_mutation_factor(iteration, best_fitness):
            if iteration > 0 and len(self.mutation_factor_history) > 1:
                if best_fitness < np.mean(self.mutation_factor_history):
                    self.mutation_factor *= 1.1
                else:
                    self.mutation_factor *= 0.9
            self.mutation_factor_history.append(self.mutation_factor)

        def update_crossover_prob(iteration, best_fitness):
            if iteration > 0 and len(self.crossover_prob_history) > 1:
                if best_fitness < np.mean(self.crossover_prob_history):
                    self.crossover_prob = min(0.9, self.crossover_prob + 0.1)
                else:
                    self.crossover_prob = max(0.1, self.crossover_prob - 0.1)
            self.crossover_prob_history.append(self.crossover_prob)

        for iteration in range(self.budget):
            par, bandwidth = update_parameters(iteration)
            update_mutation_factor(iteration, best_fitness)
            update_crossover_prob(iteration, best_fitness)

            # remaining code as before