import numpy as np

class EnhancedAntColonyOptimization:
    def __init__(self, budget, dim, colony_size=10, evaporation_rate=0.5, alpha=1.0, beta=2.0, init_pheromone=0.1, adaptive_update=True):
        self.budget = budget
        self.dim = dim
        self.colony_size = colony_size
        self.evaporation_rate = evaporation_rate
        self.alpha = alpha
        self.beta = beta
        self.init_pheromone = init_pheromone
        self.adaptive_update = adaptive_update

    def __call__(self, func):
        def evaluate_solution(solution):
            return func(solution)

        def initialize_colony():
            return np.random.uniform(-5.0, 5.0, (self.colony_size, self.dim))

        def update_pheromones(colony, pheromones):
            if self.adaptive_update:
                pheromones *= (1 - self.evaporation_rate)  # Adaptive evaporation rate
            else:
                pheromones *= self.evaporation_rate

            for ant_solution in colony:
                pheromones += self.init_pheromone / (1.0 + evaluate_solution(ant_solution))

            return pheromones

        pheromones = np.full(self.dim, self.init_pheromone)

        best_solution = None
        best_fitness = np.inf

        colony = initialize_colony()
        for _ in range(self.budget // self.colony_size):
            for ant_solution in colony:
                fitness = evaluate_solution(ant_solution)
                if fitness < best_fitness:
                    best_solution = ant_solution
                    best_fitness = fitness

            pheromones = update_pheromones(colony, pheromones)
            colony = np.random.choice(np.linspace(-5.0, 5.0, num=100), size=(self.colony_size, self.dim), p=pheromones / np.sum(pheromones))

        return best_solution