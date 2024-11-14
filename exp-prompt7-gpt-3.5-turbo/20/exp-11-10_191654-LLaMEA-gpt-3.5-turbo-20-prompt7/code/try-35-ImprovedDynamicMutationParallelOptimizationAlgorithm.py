import numpy as np

class ImprovedDynamicMutationParallelOptimizationAlgorithm(DynamicMutationParallelOptimizationAlgorithm):
    def __call__(self, func):
        best_solution = np.random.uniform(-5.0, 5.0, self.dim)
        best_fitness = func(best_solution)
        step_size = self.initial_step_size
        min_candidates = 2
        max_candidates = 10
        
        for _ in range(self.budget // min_candidates):
            candidate_solutions = [np.random.uniform(-5.0, 5.0, self.dim) for _ in range(self.num_candidates)]
            candidate_fitness = [func(candidate) for candidate in candidate_solutions]

            if min(candidate_fitness) < best_fitness:
                best_fitness = min(candidate_fitness)
                best_solution = candidate_solutions[np.argmin(candidate_fitness)]
                step_size *= 0.9

            if len(candidate_solutions) > min_candidates and len(candidate_solutions) < max_candidates:
                if np.random.rand() < 0.5:
                    candidate_solutions.pop()  # Reduce population size randomly
                else:
                    candidate_solutions += [np.random.uniform(-5.0, 5.0, self.dim)]  # Introduce new candidate randomly

            for i, candidate in enumerate(candidate_solutions):
                candidate_solutions[i] = candidate + step_size * np.random.uniform(-1, 1, self.dim)

        return best_solution