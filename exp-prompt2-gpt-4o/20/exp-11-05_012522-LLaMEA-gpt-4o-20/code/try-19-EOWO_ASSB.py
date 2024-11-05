import numpy as np

class EOWO_ASSB:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.exploration_factor = 0.2
        self.population_size = min(30, self.budget // 5)
        self.whales = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        self.best_solution = None
        self.best_fitness = float('inf')
    
    def oppositional_solution(self, solution):
        return self.lower_bound + self.upper_bound - solution

    def adaptive_boundaries(self, evaluations):
        factor = evaluations / self.budget
        lb = self.lower_bound * (1 - factor * self.exploration_factor)
        ub = self.upper_bound * (1 + factor * self.exploration_factor)
        return lb, ub

    def enhance_opposition(self, current, best):
        return current + np.random.rand(self.dim) * (best - current)

    def __call__(self, func):
        evaluations = 0

        while evaluations < self.budget:
            # Calculate fitness for current population
            fitness = np.array([func(whale) for whale in self.whales])
            evaluations += self.population_size

            # Update best solution found
            min_fitness_idx = np.argmin(fitness)
            if fitness[min_fitness_idx] < self.best_fitness:
                self.best_fitness = fitness[min_fitness_idx]
                self.best_solution = self.whales[min_fitness_idx].copy()
            
            # Update whales with adaptive boundaries
            lb, ub = self.adaptive_boundaries(evaluations)
            
            for i in range(self.population_size):
                if evaluations >= self.budget:
                    break
                if np.random.rand() < 0.5:
                    # Enhanced oppositional learning
                    enhanced_opp = self.enhance_opposition(self.whales[i], self.best_solution)
                    D = np.abs(np.random.rand(self.dim) * enhanced_opp - self.whales[i])
                    A = 2 * np.random.rand(self.dim) - 1
                    self.whales[i] = enhanced_opp - A * D
                else:
                    # Update using best solution
                    D = np.abs(np.random.rand(self.dim) * self.best_solution - self.whales[i])
                    A = 2 * np.random.rand(self.dim) - 1
                    self.whales[i] = self.best_solution - A * D
                
                # Ensure adaptive search space boundaries
                self.whales[i] = np.clip(self.whales[i], lb, ub)

        return self.best_solution, self.best_fitness