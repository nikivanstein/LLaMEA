import numpy as np

class PSOTabuSearch:
    def __init__(self, budget, dim, tabu_tenure=5):
        self.budget = budget
        self.dim = dim
        self.tabu_tenure = tabu_tenure

    def __call__(self, func):
        best_solution = np.random.uniform(-5.0, 5.0, self.dim)
        best_fitness = func(best_solution)
        tabu_list = []

        for _ in range(self.budget):
            new_solution = best_solution + np.random.uniform(-1, 1, self.dim)
            new_solution = np.clip(new_solution, -5.0, 5.0)
            new_fitness = func(new_solution)

            if new_fitness < best_fitness and new_solution.tolist() not in tabu_list:
                best_solution = new_solution
                best_fitness = new_fitness
                tabu_list.append(new_solution.tolist())
                if len(tabu_list) > self.tabu_tenure:
                    tabu_list.pop(0)

        return best_solution