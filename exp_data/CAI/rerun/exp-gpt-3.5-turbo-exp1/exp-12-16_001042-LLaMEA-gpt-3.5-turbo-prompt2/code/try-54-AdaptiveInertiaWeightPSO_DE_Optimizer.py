class AdaptiveInertiaWeightPSO_DE_Optimizer(ImprovedEnhancedExplorationMutationPSO_DE_Optimizer):
    def __call__(self, func):
        inertia_weights = np.linspace(0.9, 0.4, self.budget)
        
        for t in range(self.budget):
            self.w = inertia_weights[t]
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
                
                if np.random.rand() < 0.1:
                    adaptive_opposite_sol = np.mean(np.array([pbest[i], gbest]), axis=0) - np.random.uniform(-1, 1, size=self.dim) * np.abs(pbest[i] - gbest) / np.sqrt(self.dim)
                    adaptive_opposite_score = evaluate(adaptive_opposite_sol)
                    if adaptive_opposite_score < pbest_scores[i]:
                        pbest[i] = adaptive_opposite_sol
                        pbest_scores[i] = adaptive_opposite_score
                        if adaptive_opposite_score < gbest_score:
                            gbest = adaptive_opposite_sol
                            gbest_score = adaptive_opposite_score
        
        return gbest