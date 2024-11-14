import numpy as np

class EnhancedLevyHybridPSOSA(HybridPSOSA):
    def __init__(self, budget, dim):
        super().__init__(budget, dim)
    
    def __call__(self, func):
        def levy_flight(dim):
            beta = 1.5
            sigma = (gamma(1 + beta) * np.sin(np.pi * beta / 2) / (gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (1 / beta)
            u = np.random.normal(0, sigma, dim)
            v = np.random.normal(0, 1, dim)
            step = u / np.abs(v) ** (1 / beta)
            return 0.01 * step
        
        def pso_search(best_solution, inertia_weight):
            mutation_rate = np.random.uniform(0.1, 1.0) ** 2
            new_solution = best_solution + mutation_rate * np.random.uniform(-1.0, 1.0, self.dim) * inertia_weight
            return new_solution + levy_flight(self.dim)
        
        def sa_search(best_solution):
            mutation_rate = np.random.uniform(0.1, 1.0) ** 2
            new_solution = best_solution + mutation_rate * np.random.normal(0, 1.0, self.dim)
            return new_solution + levy_flight(self.dim)
        
        best_solution = np.random.uniform(-5.0, 5.0, self.dim)
        best_fitness = func(best_solution)
        inertia_weight = 0.5  # Initial inertia weight
        
        for _ in range(self.budget):
            new_solution = pso_search(best_solution, inertia_weight) if np.random.rand() < 0.5 else sa_search(best_solution)
            new_fitness = func(new_solution)
            
            if new_fitness < best_fitness:
                best_solution = new_solution
                best_fitness = new_fitness
                inertia_weight = max(0.4, inertia_weight * 0.99)  # Update inertia weight dynamically
        
        return best_solution