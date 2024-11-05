import numpy as np

class Enhanced_AOWO_DR:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = min(30, self.budget // 5)
        
        # Use chaos-based initialization for better distribution
        logistic_map = np.random.rand(self.population_size)
        for i in range(100):  # Iterate to chaos
            logistic_map = 4.0 * logistic_map * (1.0 - logistic_map)
        self.whales = self.lower_bound + (self.upper_bound - self.lower_bound) * logistic_map[:, None]
        self.whales = self.whales[:, :self.dim]  # Ensure dimension consistency
        
        self.best_solution = None
        self.best_fitness = float('inf')
    
    def oppositional_solution(self, solution):
        return self.lower_bound + self.upper_bound - solution

    def reduce_dimensionality(self, solution, factor):
        mask = np.random.rand(self.dim) < factor
        reduced_solution = solution.copy()
        reduced_solution[mask] = self.best_solution[mask] if self.best_solution is not None else 0
        return reduced_solution

    def __call__(self, func):
        evaluations = 0

        while evaluations < self.budget:
            fitness = np.array([func(whale) for whale in self.whales])
            evaluations += self.population_size

            min_fitness_idx = np.argmin(fitness)
            if fitness[min_fitness_idx] < self.best_fitness:
                self.best_fitness = fitness[min_fitness_idx]
                self.best_solution = self.whales[min_fitness_idx].copy()
            
            reduction_factor = 1 - (evaluations / self.budget)

            # Introduce adaptive inertia weight for better exploration-exploitation trade-off
            inertia_weight = 0.5 + (0.5 * (1 - evaluations / self.budget))
            
            for i in range(self.population_size):
                if evaluations >= self.budget:
                    break
                if np.random.rand() < 0.5:
                    D = np.abs(np.random.rand(self.dim) * self.best_solution - self.whales[i])
                    A = 2 * np.random.rand(self.dim) - 1
                    # Apply inertia weight
                    self.whales[i] = inertia_weight * (self.best_solution - A * D)
                else:
                    opp_solution = self.oppositional_solution(self.whales[i])
                    D = np.abs(np.random.rand(self.dim) * opp_solution - self.whales[i])
                    A = 2 * np.random.rand(self.dim) - 1
                    # Apply inertia weight
                    self.whales[i] = inertia_weight * (opp_solution - A * D)

                self.whales[i] = self.reduce_dimensionality(self.whales[i], reduction_factor)
                self.whales[i] = np.clip(self.whales[i], self.lower_bound, self.upper_bound)

        return self.best_solution, self.best_fitness