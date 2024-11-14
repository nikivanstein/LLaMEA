import numpy as np

class ParallelTournamentSelectionOptimizationAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.num_candidates = int(budget * 0.2)  # Modify to create multiple candidate solutions in parallel
        self.initial_step_size = 0.1

    def __call__(self, func):
        best_solution = np.random.uniform(-5.0, 5.0, self.dim)
        best_fitness = func(best_solution)
        step_size = self.initial_step_size
        
        for _ in range(self.budget // self.num_candidates):
            candidate_solutions = [np.random.uniform(-5.0, 5.0, self.dim) for _ in range(self.num_candidates)]  # Generate multiple candidate solutions concurrently
            candidate_fitness = [func(candidate) for candidate in candidate_solutions]

            best_candidate_index = np.argmin(candidate_fitness)
            if candidate_fitness[best_candidate_index] < best_fitness:
                best_fitness = candidate_fitness[best_candidate_index]
                best_solution = candidate_solutions[best_candidate_index]
                step_size *= 0.9  # Adjust step size based on performance
                
                for i, candidate in enumerate(candidate_solutions):
                    if i != best_candidate_index:
                        candidate_solutions[i] = candidate + step_size * np.random.uniform(-1, 1, self.dim)  # Dynamic mutation of candidate solutions

        return best_solution