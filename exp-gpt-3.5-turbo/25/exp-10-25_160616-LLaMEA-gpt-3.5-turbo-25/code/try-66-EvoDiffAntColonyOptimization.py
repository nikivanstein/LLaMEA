import numpy as np

class EvoDiffAntColonyOptimization:
    def __init__(self, budget, dim, colony_size=10, evaporation_rate=0.5, alpha=1.0, beta=2.0, pheromone_update_strategy='adaptive', diff_weight=0.5, diff_cross_prob=0.9):
        self.budget = budget
        self.dim = dim
        self.colony_size = colony_size
        self.evaporation_rate = evaporation_rate
        self.alpha = alpha
        self.beta = beta
        self.pheromone_update_strategy = pheromone_update_strategy
        self.diff_weight = diff_weight
        self.diff_cross_prob = diff_cross_prob

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
            for ant_idx, ant_solution in enumerate(colony):
                rand_ant_index = np.random.choice([idx for idx in range(self.colony_size) if idx != ant_idx])
                diff_vector = self.diff_weight * (colony[rand_ant_index] - ant_solution)
                mutated_solution = ant_solution + np.where(np.random.rand(self.dim) < self.diff_cross_prob, diff_vector, 0)
                colony[ant_idx] = np.clip(mutated_solution, -5.0, 5.0)

        return best_solution