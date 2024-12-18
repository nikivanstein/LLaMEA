import multiprocessing

class ParallelDynamicMutationHybridFireAntCuckooOptimization(DynamicMutationHybridFireAntCuckooOptimization):
    def __call__(self, func):
        best_solution = np.random.uniform(self.lower_bound, self.upper_bound, self.dim)
        best_fitness = func(best_solution)
        mutation_prob = 0.2
        pool = multiprocessing.Pool()

        for _ in range(self.budget):
            if np.random.rand() < mutation_prob:  
                best_solution = np.clip(best_solution + np.random.normal(0, 0.1, self.dim), self.lower_bound, self.upper_bound)
                best_fitness = func(best_solution)
                mutation_prob *= 0.99  
                self.step_size *= 0.99  # Adjust step size based on fitness improvement rate
                
            steps = self.step_size * np.random.uniform(-1, 1, (self.dim, self.dim))
            new_solutions = np.clip(best_solution + steps, self.lower_bound, self.upper_bound)
            new_fitnesses = pool.map(func, new_solutions)
            
            min_idx = np.argmin(new_fitnesses)
            if new_fitnesses[min_idx] < best_fitness:
                best_solution = new_solutions[min_idx]
                best_fitness = new_fitnesses[min_idx]
                self.step_size *= 0.98  # Dynamic step size adjustment for faster convergence
                
            cuckoo_solution = np.clip(best_solution + np.random.uniform(-0.1, 0.1, self.dim), self.lower_bound, self.upper_bound)
            cuckoo_fitness = func(cuckoo_solution)
            if cuckoo_fitness < best_fitness:
                best_solution = cuckoo_solution
                best_fitness = cuckoo_fitness
        
        pool.close()
        pool.join()
        
        return best_solution