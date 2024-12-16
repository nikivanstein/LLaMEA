class EnhancedOppositionBasedExplorationPSO_DE_Optimizer(DynamicMutationOppositionBasedExplorationPSO_DE_Optimizer):
    def __call__(self, func):
        def opposition_mutation(x, pop, mutation_factor):
            opposite_pop = np.mean(pop, axis=0) + np.random.uniform(-1, 1, size=self.dim) * np.abs(pop - x) / np.sqrt(self.dim)
            mutant = x + mutation_factor * (opposite_pop - x)
            return mutant

        population = np.random.uniform(-5.0, 5.0, size=(self.swarm_size, self.dim))
        ...
        for _ in range(self.budget):
            ...
            for i in range(self.swarm_size):
                ...
                new_sol = de(population[i], pbest[[i, (i+1)%self.swarm_size, (i+2)%self.swarm_size]], mutation_factor)
                ...
                if np.random.rand() < 0.1:
                    opposite_sol = opposition_mutation(pbest[i], gbest, mutation_factor)
                    opposite_score = evaluate(opposite_sol)
                    if opposite_score < pbest_scores[i]:
                        pbest[i] = opposite_sol
                        pbest_scores[i] = opposite_score
                        if opposite_score < gbest_score:
                            gbest = opposite_sol
                            gbest_score = opposite_score
                ...
        return gbest