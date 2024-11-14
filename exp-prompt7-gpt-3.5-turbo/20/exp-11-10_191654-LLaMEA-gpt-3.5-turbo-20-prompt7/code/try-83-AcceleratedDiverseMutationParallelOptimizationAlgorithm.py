import numpy as np

class AcceleratedDiverseMutationParallelOptimizationAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.num_candidates = int(budget * 0.2)
        self.initial_step_size = 0.1

    def __call__(self, func):
        best_solution = np.random.uniform(-5.0, 5.0, self.dim)
        best_fitness = func(best_solution)
        step_size = self.initial_step_size
        
        for _ in range(self.budget // self.num_candidates):
            candidate_solutions = [np.random.uniform(-5.0, 5.0, self.dim) for _ in range(self.num_candidates)]
            candidate_fitness = [func(candidate) for candidate in candidate_solutions]

            if min(candidate_fitness) < best_fitness:
                best_fitness = min(candidate_fitness)
                best_solution = candidate_solutions[np.argmin(candidate_fitness)]
                step_size *= 0.9
                
            for i, candidate in enumerate(candidate_solutions):
                mutation_direction = np.random.standard_cauchy(self.dim)
                candidate_solutions[i] = candidate + step_size * mutation_direction

            improvement_ratio = (best_fitness - min(candidate_fitness)) / max(1e-6, best_fitness)  # Calculate fitness improvement ratio
            step_size *= 1 + 0.1 * improvement_ratio  # Adjust step size based on fitness improvement

        return best_solution