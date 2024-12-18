from concurrent.futures import ThreadPoolExecutor

class ParallelDynamicPopulationSizeHybridFireAntCuckooOptimization(DynamicPopulationSizeHybridFireAntCuckooOptimization):
    def __call__(self, func):
        best_solution = np.random.uniform(self.lower_bound, self.upper_bound, self.dim)
        best_fitness = func(best_solution)
        mutation_prob = 0.2
        population_size = 10
        
        with ThreadPoolExecutor() as executor:
            futures = []
            for _ in range(self.budget):
                if np.random.rand() < mutation_prob:  
                    best_solution = np.clip(best_solution + np.random.normal(0, 0.1, self.dim), self.lower_bound, self.upper_bound)
                    best_fitness = func(best_solution)
                    mutation_prob *= 0.99  
                    self.step_size *= 0.99  # Adjust step size based on fitness improvement rate

                steps = self.step_size * np.random.uniform(-1, 1, (population_size, self.dim))
                
                for step in steps:
                    new_solution = np.clip(best_solution + step, self.lower_bound, self.upper_bound)
                    futures.append(executor.submit(func, new_solution))
                    
                results = [future.result() for future in futures]
                new_fitnesses = np.array(results)
                
                min_idx = np.argmin(new_fitnesses)
                if new_fitnesses[min_idx] < best_fitness:
                    best_solution = best_solution + steps[min_idx]
                    best_fitness = new_fitnesses[min_idx]
                    self.step_size *= 0.98  # Dynamic step size adjustment for faster convergence

                cuckoo_solution = np.clip(best_solution + np.random.uniform(-0.1, 0.1, self.dim), self.lower_bound, self.upper_bound)
                cuckoo_fitness = func(cuckoo_solution)
                if cuckoo_fitness < best_fitness:
                    best_solution = cuckoo_solution
                    best_fitness = cuckoo_fitness

        return best_solution