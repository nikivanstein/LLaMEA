import concurrent.futures

class ParallelDynamicMutationHybridFireAntCuckooOptimization(DynamicMutationHybridFireAntCuckooOptimization):
    def __call__(self, func):
        best_solution = np.random.uniform(self.lower_bound, self.upper_bound, self.dim)
        best_fitness = func(best_solution)
        mutation_prob = 0.2
        num_threads = 4
        
        def evaluate_solution(solution):
            fitness = func(solution)
            return solution, fitness
        
        with concurrent.futures.ThreadPoolExecutor() as executor:
            for _ in range(self.budget // num_threads):
                solutions = [np.clip(best_solution + np.random.normal(0, 0.1, self.dim), self.lower_bound, self.upper_bound) for _ in range(num_threads)]
                results = list(executor.map(evaluate_solution, solutions))
                
                for sol, fit in results:
                    if fit < best_fitness:
                        best_solution = sol
                        best_fitness = fit
                        self.step_size *= 0.98
                        
                for _ in range(num_threads):
                    steps = self.step_size * np.random.uniform(-1, 1, (self.dim, self.dim))
                    new_solutions = np.clip(best_solution + steps, self.lower_bound, self.upper_bound)
                    new_fitnesses = np.array([func(sol) for sol in new_solutions])
                    
                    min_idx = np.argmin(new_fitnesses)
                    if new_fitnesses[min_idx] < best_fitness:
                        best_solution = new_solutions[min_idx]
                        best_fitness = new_fitnesses[min_idx]
                        self.step_size *= 0.98
                    
                    cuckoo_solution = np.clip(best_solution + np.random.uniform(-0.1, 0.1, self.dim), self.lower_bound, self.upper_bound)
                    cuckoo_fitness = func(cuckoo_solution)
                    if cuckoo_fitness < best_fitness:
                        best_solution = cuckoo_solution
                        best_fitness = cuckoo_fitness
        
        return best_solution