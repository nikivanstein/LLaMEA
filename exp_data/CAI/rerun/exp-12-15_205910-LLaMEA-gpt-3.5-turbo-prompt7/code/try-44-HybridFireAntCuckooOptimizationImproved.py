import numpy as np

class HybridFireAntCuckooOptimizationImproved(FireAntOptimizationMutationImproved):
    def __call__(self, func):
        best_solution = np.random.uniform(self.lower_bound, self.upper_bound, self.dim)
        best_fitness = func(best_solution)
        fitness_history = []
        
        for _ in range(self.budget):
            if np.random.rand() < 0.2:  # Introduce random mutation with 20% probability
                best_solution = np.clip(best_solution + np.random.normal(0, 0.1, self.dim), self.lower_bound, self.upper_bound)
                best_fitness = func(best_solution)
            
            steps = self.step_size * np.random.uniform(-1, 1, (self.dim, self.dim))
            new_solutions = np.clip(best_solution + steps, self.lower_bound, self.upper_bound)
            new_fitnesses = np.array([func(sol) for sol in new_solutions])
            
            min_idx = np.argmin(new_fitnesses)
            if new_fitnesses[min_idx] < best_fitness:
                best_solution = new_solutions[min_idx]
                best_fitness = new_fitnesses[min_idx]
                fitness_history.append(1)  # Keep track of fitness improvement
                self.step_size *= 0.99
                if _ % 100 == 0 and len(fitness_history) > 10:  # Dynamic step size adaptation based on recent fitness improvements
                    self.step_size *= 1.01 if sum(fitness_history[-10:]) >= 7 else 1.005
            else:
                fitness_history.append(0)
            
            # Cuckoo search exploration
            cuckoo_solution = np.clip(best_solution + np.random.uniform(-0.1, 0.1, self.dim), self.lower_bound, self.upper_bound)
            cuckoo_fitness = func(cuckoo_solution)
            if cuckoo_fitness < best_fitness:
                best_solution = cuckoo_solution
                best_fitness = cuckoo_fitness
        
        return best_solution