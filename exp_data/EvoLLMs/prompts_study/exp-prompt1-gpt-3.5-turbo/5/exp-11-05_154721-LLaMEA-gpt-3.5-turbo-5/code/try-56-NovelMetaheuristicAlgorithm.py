class NovelMetaheuristicAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.mutation_scale = 0.5  # Initialize mutation scale

    def __call__(self, func):
        best_solution = np.random.uniform(-5.0, 5.0, self.dim)  # Initialize random solution
        best_fitness = func(best_solution)
        
        for eval_count in range(self.budget):
            if eval_count % (self.budget // 10) == 0 and eval_count > 0:
                self.mutation_scale = 0.5 - 0.5 * eval_count / self.budget  # Adapt mutation scale

            gaussian_mutation = np.random.normal(0, self.mutation_scale, self.dim)  # Gaussian mutation
            cauchy_mutation = np.random.standard_cauchy(self.dim) * 0.1  # Cauchy mutation
            candidate_solution = best_solution + (gaussian_mutation + cauchy_mutation)  # Combined mutation
            candidate_solution = np.clip(candidate_solution, -5.0, 5.0)  # Ensure solution is within bounds
            candidate_fitness = func(candidate_solution)
            
            if candidate_fitness < best_fitness:
                best_solution = candidate_solution
                best_fitness = candidate_fitness
        
        return best_solution