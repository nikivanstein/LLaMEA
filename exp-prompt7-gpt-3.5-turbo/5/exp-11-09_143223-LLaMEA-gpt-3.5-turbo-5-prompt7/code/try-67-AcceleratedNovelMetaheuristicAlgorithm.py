import numpy as np

class AcceleratedNovelMetaheuristicAlgorithm(NovelMetaheuristicAlgorithm):
    def __init__(self, budget, dim):
        super().__init__(budget, dim)
        self.num_steps = 5  # Number of mutation steps
        
    def __call__(self, func):
        best_solution = np.random.uniform(-5.0, 5.0, self.dim)
        best_fitness = func(best_solution)
        
        for _ in range(self.budget):
            candidate_solution = best_solution.copy()
            for _ in range(self.num_steps):
                mutation_step = self.mutation_step * np.random.uniform(0.5, 1.5)
                candidate_solution += mutation_step * np.random.uniform(-1, 1, self.dim)
                candidate_solution = np.clip(candidate_solution, -5.0, 5.0)
                candidate_fitness = func(candidate_solution)
            
                if candidate_fitness < best_fitness:
                    best_solution = candidate_solution
                    best_fitness = candidate_fitness
            
            if np.random.rand() < 0.1:
                self.mutation_step *= np.exp(0.1 * np.random.uniform(-1, 1))
                self.mutation_step = max(0.1, min(self.mutation_step, 2.0))

        return best_solution