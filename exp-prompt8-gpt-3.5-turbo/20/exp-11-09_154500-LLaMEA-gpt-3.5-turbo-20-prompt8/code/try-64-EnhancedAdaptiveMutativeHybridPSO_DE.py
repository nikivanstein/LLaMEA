class EnhancedAdaptiveMutativeHybridPSO_DE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.w_min = 0.4
        self.w_max = 0.9
        self.c1 = 1.49445
        self.c2 = 1.49445
        self.pop_size = 20
        self.max_velocity = 0.1 * (5.0 - (-5.0))
        self.base_mutation_scale = 0.5
        self.mutation_scale = self.base_mutation_scale
        self.history_fitness = []
        self.previous_best_fitness = np.inf

    def __call__(self, func):
        population = np.random.uniform(-5.0, 5.0, (self.pop_size, self.dim))
        velocity = np.zeros((self.pop_size, self.dim))
        pbest = population.copy()
        pbest_fitness = np.array([func(ind) for ind in population])
        gbest_idx = np.argmin(pbest_fitness)
        gbest = pbest[gbest_idx]

        for t in range(1, self.budget + 1):
            improved = np.mean(self.previous_best_fitness) < np.mean(pbest_fitness)
            self.previous_best_fitness = pbest_fitness.copy()
            w = self.w_min + (self.w_max - self.w_min) * (1 - t / self.budget) ** 2 if not improved else self.w_max

            r1, r2 = np.random.rand(), np.random.rand()

            # PSO update
            velocity = w * velocity + self.c1 * r1 * (pbest - population) + self.c2 * r2 * (np.tile(gbest, (self.pop_size, 1)) - population)
            velocity = np.clip(velocity, -self.max_velocity, self.max_velocity)
            population += velocity

            # Adaptive mutation based on historical fitness
            diversity = np.std(population, axis=0)
            if len(self.history_fitness) > 0:
                avg_fitness = np.mean(self.history_fitness)
                mutation_scale = self.base_mutation_scale * (1 - t / self.budget) + np.random.normal(0, 0.1) * (diversity / (avg_fitness + 1e-10))
            else:
                mutation_scale = self.base_mutation_scale * (1 - t / self.budget)
            mutant = population + np.random.uniform(-mutation_scale, mutation_scale, (self.pop_size, self.dim)) * (population - population[np.random.randint(self.pop_size, size=self.pop_size)])

            fitness = np.array([func(ind) for ind in mutant])
            update_indices = fitness < pbest_fitness
            pbest[update_indices] = mutant[update_indices]
            pbest_fitness[update_indices] = fitness[update_indices]

            gbest_idx = np.argmin(pbest_fitness)
            gbest = pbest[gbest_idx]

            self.history_fitness.append(fitness.mean())

        return gbest