import numpy as np

class ImprovedDynamicMutationParallelOptimizationAlgorithm:
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

            ranked_candidates = [c for _, c in sorted(zip(candidate_fitness, candidate_solutions))]  # Sort candidates based on fitness

            selected_idx = np.random.choice(range(self.num_candidates), p=[1 - i/self.num_candidates for i in range(self.num_candidates)])  # Roulette wheel selection
            if func(ranked_candidates[selected_idx]) < best_fitness:
                best_fitness = func(ranked_candidates[selected_idx])
                best_solution = np.copy(ranked_candidates[selected_idx])
                step_size *= 0.9  # Adjust step size based on performance
                
                for i, candidate in enumerate(candidate_solutions):
                    candidate_solutions[i] = candidate + step_size * np.random.uniform(-1, 1, self.dim)  # Dynamic mutation of candidate solutions

        return best_solution