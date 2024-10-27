import numpy as np

class DifferentialEvolutionEnhancedACO:
    def __init__(self, budget, dim, colony_size=10, evaporation_rate=0.5, alpha=1.0, beta=2.0, pheromone_update_strategy='adaptive', de_cr=0.9, de_f=0.5):
        self.budget = budget
        self.dim = dim
        self.colony_size = colony_size
        self.evaporation_rate = evaporation_rate
        self.alpha = alpha
        self.beta = beta
        self.pheromone_update_strategy = pheromone_update_strategy
        self.de_cr = de_cr
        self.de_f = de_f

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

        def differential_evolution(colony):
            for i, ant_solution in enumerate(colony):
                r1, r2, r3 = np.random.choice(len(colony), 3, replace=False)
                mutant = colony[r1] + self.de_f * (colony[r2] - colony[r3])
                cross_points = np.random.rand(self.dim) < self.de_cr
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                trial = np.where(cross_points, mutant, ant_solution)
                if evaluate_solution(trial) < evaluate_solution(ant_solution):
                    colony[i] = trial

            return colony

        best_solution = None
        best_fitness = np.inf
        pheromones = np.ones(self.dim)

        colony = initialize_colony()
        for _ in range(self.budget // self.colony_size):
            colony = differential_evolution(colony)
            for ant_solution in colony:
                fitness = evaluate_solution(ant_solution)
                if fitness < best_fitness:
                    best_solution = ant_solution
                    best_fitness = fitness

            pheromones = update_pheromones(colony, pheromones, _)
            colony = np.array([best_solution + np.random.randn(self.dim) for _ in range(self.colony_size)])

        return best_solution