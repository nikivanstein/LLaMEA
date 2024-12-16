class EnhancedDynamicMutationOppositionBasedExplorationPSO_DE_Optimizer(DynamicMutationOppositionBasedExplorationPSO_DE_Optimizer):
    def __call__(self, func):
        def de(x, pop, F):
            idxs = np.random.choice(len(pop), 3, replace=False)
            a, b, c = pop[idxs]
            mutant = a + F * (b - c)
            crossover = np.random.rand(self.dim) < self.crossover_prob
            trial = np.where(crossover, mutant, x)
            return trial
        
        def evaluate(x):
            return func(x)
        
        population = np.random.uniform(-5.0, 5.0, size=(self.swarm_size, self.dim))
        velocities = np.zeros((self.swarm_size, self.dim))
        pbest = population.copy()
        pbest_scores = np.array([evaluate(p) for p in pbest])
        gbest_idx = np.argmin(pbest_scores)
        gbest = pbest[gbest_idx]
        gbest_score = pbest_scores[gbest_idx]
        mutation_factors = np.full(self.swarm_size, self.mutation_factor)
        
        for _ in range(self.budget):
            r1 = np.random.rand(self.swarm_size, self.dim)
            r2 = np.random.rand(self.swarm_size, self.dim)
            velocities = self.w * velocities + self.c1 * r1 * (pbest - population) + self.c2 * r2 * (gbest - population)
            population += velocities
            
            for i in range(self.swarm_size):
                mutation_factor = np.clip(np.random.normal(mutation_factors[i], 0.1), 0.1, 0.9)  # Dynamic mutation factor
                new_sol = de(population[i], pbest[[i, (i+1)%self.swarm_size, (i+2)%self.swarm_size]], mutation_factor)
                new_score = evaluate(new_sol)
                if new_score < pbest_scores[i]:
                    pbest[i] = new_sol
                    pbest_scores[i] = new_score
                    if new_score < gbest_score:
                        gbest = new_sol
                        gbest_score = new_score
                
                if np.random.rand() < 0.1:
                    opposite_sol = np.mean(np.array([pbest[i], gbest]), axis=0) + np.random.uniform(-1, 1, size=self.dim) * np.abs(pbest[i] - gbest) / np.sqrt(self.dim)
                    opposite_score = evaluate(opposite_sol)
                    if opposite_score < pbest_scores[i]:
                        pbest[i] = opposite_sol
                        pbest_scores[i] = opposite_score
                        if opposite_score < gbest_score:
                            gbest = opposite_sol
                            gbest_score = opposite_score
                            
                mutation_factors[i] = mutation_factor  # Update mutation factor
        
        return gbest