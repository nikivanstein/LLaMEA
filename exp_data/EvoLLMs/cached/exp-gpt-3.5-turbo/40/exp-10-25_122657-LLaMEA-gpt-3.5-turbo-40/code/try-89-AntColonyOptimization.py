import numpy as np

class AntColonyOptimization:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.num_ants = 10

    def __call__(self, func):
        def initialize_colony():
            return np.random.uniform(self.lower_bound, self.upper_bound, (self.num_ants, self.dim))

        def evaluate_colony(colony):
            return np.array([func(solution) for solution in colony])

        def update_colony(colony, pheromones):
            best_idx = np.argmin(pheromones)
            best_solution = colony[best_idx]

            for i in range(self.num_ants):
                if i != best_idx:
                    new_solution = colony[i] + np.random.normal(0, 1, self.dim) * (best_solution - colony[i])
                    new_solution = np.clip(new_solution, self.lower_bound, self.upper_bound)
                    if func(new_solution) < pheromones[i]:
                        colony[i] = new_solution
                        pheromones[i] = func(new_solution)

            return colony, pheromones

        colony = initialize_colony()
        pheromones = evaluate_colony(colony)

        for _ in range(self.budget - self.budget // 10):
            colony, pheromones = update_colony(colony, pheromones)

        best_idx = np.argmin(pheromones)
        return colony[best_idx]