import numpy as np

class EnhancedGreyWolfOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lb = -5.0
        self.ub = 5.0

    def __call__(self, func):
        def initialize_population():
            return np.random.uniform(self.lb, self.ub, (self.budget, self.dim))

        def optimize():
            population = initialize_population()
            alpha, beta, delta = population[np.argsort([func(ind) for ind in population])[:3]]
            
            # Integrate Simulated Annealing for exploration
            temperature = 1.0
            cooling_rate = 0.95

            for _ in range(self.budget):
                a = 2 - 2 * (_ / self.budget)
                for i in range(self.budget):
                    x = population[i]
                    X1 = alpha - a * np.abs(2 * np.random.rand(self.dim) * alpha - x)
                    X2 = beta - a * np.abs(2 * np.random.rand(self.dim) * beta - x)
                    X3 = delta - a * np.abs(2 * np.random.rand(self.dim) * delta - x)
                    population[i] = (X1 + X2 + X3) / 3

                    # Simulated Annealing
                    candidate = (X1 + X2 + X3) / 3
                    candidate_fitness = func(candidate)
                    if candidate_fitness < func(population[i]) or np.random.rand() < np.exp((func(population[i]) - candidate_fitness) / temperature):
                        population[i] = candidate

                alpha, beta, delta = population[np.argsort([func(ind) for ind in population])[:3]]
                
                # Cooling the temperature
                temperature *= cooling_rate

            return alpha

        return optimize()