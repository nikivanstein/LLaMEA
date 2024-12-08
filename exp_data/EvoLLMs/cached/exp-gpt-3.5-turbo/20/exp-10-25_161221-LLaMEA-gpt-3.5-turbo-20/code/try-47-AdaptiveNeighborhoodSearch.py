import numpy as np

class AdaptiveNeighborhoodSearch:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        def generate_neighbor(solution, neighborhood):
            return np.clip(solution + np.random.uniform(-neighborhood, neighborhood, self.dim), -5.0, 5.0)

        solution = np.random.uniform(-5.0, 5.0, self.dim)
        best_solution = solution
        best_fitness = func(solution)
        neighborhoods = [0.1, 0.5, 1.0]

        for _ in range(self.budget):
            for neighborhood in neighborhoods:
                neighbor = generate_neighbor(solution, neighborhood)
                neighbor_fitness = func(neighbor)
                if neighbor_fitness < best_fitness:
                    best_solution = neighbor
                    best_fitness = neighbor_fitness
            best_neighbors = np.argsort([func(generate_neighbor(solution, n)) for n in neighborhoods])
            neighborhoods = np.clip(np.array(neighborhoods) * (1 + (0.1 if best_neighbors[0] > 1 else -0.1)), 0.1, 1.0)
            solution = best_solution

        return best_solution