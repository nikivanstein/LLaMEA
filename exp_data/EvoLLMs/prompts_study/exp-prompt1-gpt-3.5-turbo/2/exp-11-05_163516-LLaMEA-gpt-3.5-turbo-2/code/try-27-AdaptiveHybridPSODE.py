class AdaptiveHybridPSODE(HybridPSODE):
    def __init__(self, budget, dim):
        super().__init__(budget, dim)
        self.mutation_rates = np.ones(self.pop_size)
    
    def __call__(self, func):
        best_position = self.population[np.argmin([func(ind) for ind in self.population])]
        best_fitness = func(best_position)
        for _ in range(self.max_iter):
            for i in range(self.pop_size):
                # PSO update
                r1, r2 = np.random.uniform(0, 1, 2)
                new_velocity = self.w * self.population[i] + self.c1 * r1 * (best_position - self.population[i]) + self.c2 * r2 * (best_position - self.population[i])
                new_position = self.population[i] + new_velocity
                new_position = np.clip(new_position, -5.0, 5.0)
                
                # DE update
                rand_indexes = np.random.choice(np.arange(self.pop_size), 3, replace=False)
                mutant = self.population[rand_indexes[0]] + self.f * (self.population[rand_indexes[1]] - self.population[rand_indexes[2]])
                crossover = np.random.rand(self.dim) < self.cr
                trial = np.where(crossover, mutant, self.population[i])
                
                if func(trial) < func(self.population[i]):
                    self.population[i] = trial
                    self.mutation_rates[i] *= 1.2  # Adaptive mutation rate
                if func(new_position) < func(self.population[i]):
                    self.population[i] = new_position
                    self.mutation_rates[i] *= 1.2  # Adaptive mutation rate
                else:
                    self.mutation_rates[i] *= 0.9  # Adaptive mutation rate
                
                # Update the global best
                if func(self.population[i]) < best_fitness:
                    best_position = self.population[i]
                    best_fitness = func(best_position)
        
        return best_position