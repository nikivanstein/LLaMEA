class ChaoticAdaptiveDEImprovedOpposition(EnhancedAdaptiveDEImprovedOpposition):
    def __init__(self, budget, dim):
        super().__init__(budget, dim)
        
    def __call__(self, func):
        def chaotic_map(x, a=3.57, b=0.6):
            return b * x * (1 - x) if x < 0.5 else b * (1 - x) * x

        # Initialize population and best solution

        for _ in range(self.budget):
            trial_population = []
            for i in range(self.NP):
                # Mutation based on chaotic map
                chaotic_val = chaotic_map(_ / self.budget)
                mutant = population[a] + F * (population[b] - population[c]) + chaotic_val * (best_solution - population[i])

                # Crossover and selection

        return best_solution