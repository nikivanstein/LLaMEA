class DynamicMutationPSO_DE_Optimizer(ImprovedEnhancedExplorationMutationPSO_DE_Optimizer):
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
        
        for _ in range(self.budget):
            r1 = np.random.rand(self.swarm_size, self.dim)
            r2 = np.random.rand(self.swarm_size, self.dim)
            velocities = self.w * velocities + self.c1 * r1 * (pbest - population) + self.c2 * r2 * (gbest - population)
            population += velocities
            
            for i in range(self.swarm_size):
                mutation_factor = np.clip(np.random.normal(self.mutation_factor, 0.1), 0.1, 0.9)
                new_sol = de(population[i], pbest[[i, (i+1)%self.swarm_size, (i+2)%self.swarm_size]], mutation_factor)
                new_score = evaluate(new_sol)
                if new_score < pbest_scores[i]:
                    pbest[i] = new_sol
                    pbest_scores[i] = new_score
                    if new_score < gbest_score:
                        gbest = new_sol
                        gbest_score = new_score
                
                # Dynamic mutation control based on individual performance for enhanced diversity
                if np.random.rand() < 0.1:
                    mutation_factor = np.clip(np.random.normal(mutation_factor, 0.1), 0.1, 0.9)
                    dynamic_mutant = de(population[i], pbest[[i, (i+1)%self.swarm_size, (i+2)%self.swarm_size]], mutation_factor)
                    dynamic_score = evaluate(dynamic_mutant)
                    if dynamic_score < pbest_scores[i]:
                        pbest[i] = dynamic_mutant
                        pbest_scores[i] = dynamic_score
                        if dynamic_score < gbest_score:
                            gbest = dynamic_mutant
                            gbest_score = dynamic_score
        
        return gbest