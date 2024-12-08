import numpy as np

class CoevolutionaryOptimizer:
    def __init__(self, budget, dim, num_subpopulations=5, num_individuals_per_subpopulation=10):
        self.budget = budget
        self.dim = dim
        self.num_subpopulations = num_subpopulations
        self.num_individuals_per_subpopulation = num_individuals_per_subpopulation

    def __call__(self, func):
        subpopulations = [np.random.uniform(-5.0, 5.0, (self.num_individuals_per_subpopulation, self.dim)) for _ in range(self.num_subpopulations)]
        best_positions = [pop[np.argmin([func(p) for p in pop])] for pop in subpopulations]

        for _ in range(self.budget):
            for i in range(self.num_subpopulations):
                for j in range(self.num_individuals_per_subpopulation):
                    candidate_position = np.clip(subpopulations[i][j] + np.random.normal(0, 1, self.dim), -5.0, 5.0)
                    if func(candidate_position) < func(subpopulations[i][j]):
                        subpopulations[i][j] = candidate_position
                        if func(candidate_position) < func(best_positions[i]):
                            best_positions[i] = candidate_position

        return best_positions[np.argmin([func(p) for p in best_positions])]