class EnhancedOppositionBasedExplorationPSO_DE_Optimizer(DynamicMutationOppositionBasedExplorationPSO_DE_Optimizer):
    def __init__(self, budget, dim):
        super().__init__(budget, dim)
        self.alpha = 0.1
        self.beta = 0.9

    def __call__(self, func):
        def mutation_strategy(x, pbest_x, gbest_x, mutation_factor):
            return x + mutation_factor * ((pbest_x - x) + self.alpha * (gbest_x - x))

        for _ in range(self.budget):
            # Existing code for PSO-DE optimization

            for i in range(self.swarm_size):
                mutation_factor = np.clip(np.random.normal(mutation_factors[i], 0.1), 0.1, 0.9)
                new_sol = mutation_strategy(population[i], pbest[i], gbest, mutation_factor)
                new_score = evaluate(new_sol)
                if new_score < pbest_scores[i]:
                    pbest[i] = new_sol
                    pbest_scores[i] = new_score
                    if new_score < gbest_score:
                        gbest = new_sol
                        gbest_score = new_score

            # Existing code for opposition-based learning

                mutation_factors[i] = mutation_factor

        return gbest