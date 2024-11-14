import numpy as np

class DynamicMutationHybridPSOSA(HybridPSOSA):
    def __init__(self, budget, dim):
        super().__init__(budget, dim)
    
    def __call__(self, func):
        def pso_search(best_solution, inertia_weight, mutation_rate):
            new_solution = best_solution + mutation_rate * np.random.uniform(-1.0, 1.0, self.dim) * inertia_weight
            return new_solution
        
        def sa_search(best_solution, mutation_rate):
            new_solution = best_solution + mutation_rate * np.random.normal(0, 1.0, self.dim)
            return new_solution
        
        best_solution = np.random.uniform(-5.0, 5.0, self.dim)
        best_fitness = func(best_solution)
        inertia_weight = 0.5  
        mutation_rate = 0.5  # Initial mutation rate
        
        for _ in range(self.budget):
            new_solution = pso_search(best_solution, inertia_weight, mutation_rate) if np.random.rand() < 0.5 else sa_search(best_solution, mutation_rate)
            new_fitness = func(new_solution)
            
            if new_fitness < best_fitness:
                best_solution = new_solution
                best_fitness = new_fitness
                inertia_weight = max(0.4, inertia_weight * 0.99)  
                mutation_rate = max(0.1, mutation_rate * 0.98)  # Update mutation rate dynamically
        
        return best_solution