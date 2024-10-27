import numpy as np

class DynamicAntColonyOptimization:
    def __init__(self, budget, dim, colony_size=10, evaporation_rate=0.5, alpha=1.0, beta=2.0, pheromone_update_strategy='adaptive', search_radius=0.1):
        self.budget = budget
        self.dim = dim
        self.colony_size = colony_size
        self.evaporation_rate = evaporation_rate
        self.alpha = alpha
        self.beta = beta
        self.pheromone_update_strategy = pheromone_update_strategy
        self.search_radius = search_radius

    def __call__(self, func):
        def evaluate_solution(solution):
            return func(solution)

        def initialize_colony():
            return np.random.uniform(-5.0, 5.0, (self.colony_size, self.dim))

        def update_pheromones(colony, pheromones, iteration):
            if self.pheromone_update_strategy == 'adaptive':
                pheromones *= self.evaporation_rate
                pheromones += 1.0 / (1.0 + evaluate_solution(colony[np.argmin([evaluate_solution(sol) for sol in colony])]))
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
            colony = np.array([best_solution + np.random.uniform(-self.search_radius, self.search_radius, self.dim) for _ in range(self.colony_size)])

        return best_solution