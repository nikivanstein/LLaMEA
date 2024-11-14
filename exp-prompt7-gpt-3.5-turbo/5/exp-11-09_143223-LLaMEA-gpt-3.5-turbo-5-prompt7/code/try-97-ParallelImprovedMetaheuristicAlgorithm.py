from concurrent.futures import ProcessPoolExecutor

class ParallelImprovedMetaheuristicAlgorithm(ImprovedMetaheuristicAlgorithm):
    def __init__(self, budget, dim):
        super().__init__(budget, dim)

    def __call__(self, func):
        best_solution = np.random.uniform(-5.0, 5.0, self.dim)
        best_fitness = func(best_solution)
        
        with ProcessPoolExecutor() as executor:
            for _ in range(self.budget):
                future_results = [executor.submit(self.mutate_and_evaluate, func, best_solution, self.mutation_step) for _ in range(4)]
                results = [future.result() for future in future_results]
                
                for candidate_solution, candidate_fitness in results:
                    if candidate_fitness < best_fitness:
                        best_solution = candidate_solution
                        best_fitness = candidate_fitness
                        self.mutation_step *= 1.1  # Dynamic mutation step adjustment
                        self.mutation_step = max(0.1, min(self.mutation_step, 2.0))
                    else:
                        self.mutation_step *= 0.9  # Reduce mutation step if no fitness improvement
                
                self.mutation_prob = max(0.1, min(self.mutation_prob + 0.05 * np.random.uniform(-1, 1), 0.9))

        return best_solution