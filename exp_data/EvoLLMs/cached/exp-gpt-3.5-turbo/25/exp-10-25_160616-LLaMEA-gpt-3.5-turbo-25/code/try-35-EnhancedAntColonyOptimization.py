import numpy as np

class EnhancedAntColonyOptimization:
    def __init__(self, budget, dim, colony_size=10, evaporation_rate=0.5, alpha=1.0, beta=2.0, pheromone_update_strategy='dynamic', elitism_rate=0.1, local_search_prob=0.1):
        self.budget = budget
        self.dim = dim
        self.colony_size = colony_size
        self.evaporation_rate = evaporation_rate
        self.alpha = alpha
        self.beta = beta
        self.pheromone_update_strategy = pheromone_update_strategy
        self.elitism_rate = elitism_rate
        self.local_search_prob = local_search_prob

    def __call__(self, func):
        def evaluate_solution(solution):
            return func(solution)

        def initialize_colony():
            return np.random.uniform(-5.0, 5.0, (self.colony_size, self.dim))

        def update_pheromones(colony, pheromones, iteration):
            if self.pheromone_update_strategy == 'dynamic':
                pheromones *= self.evaporation_rate
            else:
                pheromones = np.ones(self.dim)

            for ant_solution in colony:
                pheromones += 1.0 / (1.0 + evaluate_solution(ant_solution))

            return pheromones

        def elitism_strategy(colony, best_solution_local):
            num_elites = max(1, int(self.elitism_rate * self.colony_size))
            elite_indices = np.argsort([evaluate_solution(solution) for solution in colony])[:num_elites]

            for idx in elite_indices:
                colony[idx] = best_solution_local

            return colony

        def local_search(solution):
            if np.random.rand() < self.local_search_prob:
                perturbation = np.random.uniform(-0.1, 0.1, size=self.dim)
                return np.clip(solution + perturbation, -5.0, 5.0)
            else:
                return solution

        best_solution = None
        best_fitness = np.inf
        pheromones = np.ones(self.dim)

        colony = initialize_colony()
        for _ in range(self.budget // self.colony_size):
            for idx, ant_solution in enumerate(colony):
                ant_solution = local_search(ant_solution)
                fitness = evaluate_solution(ant_solution)
                if fitness < best_fitness:
                    best_solution = ant_solution
                    best_fitness = fitness

                colony[idx] = ant_solution

            pheromones = update_pheromones(colony, pheromones, _)
            best_solution_local = colony[np.argmin([evaluate_solution(solution) for solution in colony])]
            colony = elitism_strategy(colony, best_solution_local)

        return best_solution