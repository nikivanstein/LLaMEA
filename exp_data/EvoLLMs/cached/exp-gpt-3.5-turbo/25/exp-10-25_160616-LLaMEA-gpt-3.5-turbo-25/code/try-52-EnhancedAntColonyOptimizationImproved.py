import numpy as np

class EnhancedAntColonyOptimizationImproved:
    def __init__(self, budget, dim, colony_size=10, evaporation_rate=0.5, alpha=1.0, beta=2.0, pheromone_update_strategy='adaptive'):
        self.budget = budget
        self.dim = dim
        self.colony_size = colony_size
        self.evaporation_rate = evaporation_rate
        self.alpha = alpha
        self.beta = beta
        self.pheromone_update_strategy = pheromone_update_strategy

    def __call__(self, func):
        def evaluate_solution(solution):
            return func(solution)

        def initialize_colony():
            return np.random.uniform(-5.0, 5.0, (self.colony_size, self.dim))

        def update_pheromones(colony, pheromones, iteration):
            if self.pheromone_update_strategy == 'adaptive':
                pheromones *= self.evaporation_rate
                elite_solution = colony[np.argmin([evaluate_solution(sol) for sol in colony])]
                pheromones += 1.0 / (1.0 + evaluate_solution(elite_solution))
                pheromones += np.random.randn(self.dim) * 0.1  # Random perturbation for exploration
            else:
                pheromones = np.ones(self.dim)

            return pheromones

        best_solution = None
        best_fitness = np.inf
        pheromones = np.ones(self.dim)

        colony = initialize_colony()
        for _ in range(self.budget // self.colony_size):
            for ant_solution in colony:
                fitness = evaluate_solution(ant_solution)
                if fitness < best_fitness:
                    best_solution = ant_solution
                    best_fitness = fitness

            pheromones = update_pheromones(colony, pheromones, _)
            colony = np.array([best_solution + np.random.randn(self.dim) for _ in range(self.colony_size)])

        return best_solution