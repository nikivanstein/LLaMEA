import numpy as np

class DynamicHybridFireAntCuckooOptimization(HybridFireAntCuckooOptimization):
    def __call__(self, func):
        best_solution = np.random.uniform(self.lower_bound, self.upper_bound, self.dim)
        best_fitness = func(best_solution)
        local_radius = 0.1  # Initial local search radius
        
        for _ in range(self.budget):
            if np.random.rand() < 0.2:  # Introduce random mutation with 20% probability
                best_solution = np.clip(best_solution + np.random.normal(0, local_radius, self.dim), self.lower_bound, self.upper_bound)
                best_fitness = func(best_solution)
            
            steps = self.step_size * np.random.uniform(-1, 1, (self.dim, self.dim))
            new_solutions = np.clip(best_solution + steps, self.lower_bound, self.upper_bound)
            new_fitnesses = np.array([func(sol) for sol in new_solutions])
            
            min_idx = np.argmin(new_fitnesses)
            if new_fitnesses[min_idx] < best_fitness:
                best_solution = new_solutions[min_idx]
                best_fitness = new_fitnesses[min_idx]
                local_radius *= 0.99  # Dynamic local search radius adaptation
                if _ % 100 == 0:
                    local_radius *= 1.01
                
            # Cuckoo search exploration
            cuckoo_solution = np.clip(best_solution + np.random.uniform(-0.1, 0.1, self.dim), self.lower_bound, self.upper_bound)
            cuckoo_fitness = func(cuckoo_solution)
            if cuckoo_fitness < best_fitness:
                best_solution = cuckoo_solution
                best_fitness = cuckoo_fitness
        
        return best_solution