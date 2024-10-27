import numpy as np

class AdaptiveNeighborhoodSearch:
    def __init__(self, budget, dim, neighborhoods=[0.1, 0.5, 1.0]):
        self.budget = budget
        self.dim = dim
        self.neighborhoods = neighborhoods

    def __call__(self, func):
        def generate_neighbor(solution, neighborhood):
            return np.clip(solution + np.random.uniform(-neighborhood, neighborhood, self.dim), -5.0, 5.0)

        solution = np.random.uniform(-5.0, 5.0, self.dim)
        best_solution = solution
        best_fitness = func(solution)

        for _ in range(self.budget):
            for neighborhood in self.neighborhoods:
                neighbor = generate_neighbor(solution, neighborhood)
                neighbor_fitness = func(neighbor)
                if neighbor_fitness < best_fitness:
                    best_solution = neighbor
                    best_fitness = neighbor_fitness
            solution = best_solution
            self.neighborhoods = [neighborhood * 0.9 if neighbor_fitness < best_fitness else neighborhood * 1.1 for neighborhood in self.neighborhoods]

        return best_solution