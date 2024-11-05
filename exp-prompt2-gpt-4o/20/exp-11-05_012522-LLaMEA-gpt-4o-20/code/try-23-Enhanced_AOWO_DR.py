import numpy as np

class Enhanced_AOWO_DR:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = min(30, self.budget // 5)
        self.whales = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        self.best_solution = None
        self.best_fitness = float('inf')
    
    def oppositional_solution(self, solution, mean_point):
        return mean_point + (mean_point - solution)

    def reduce_dimensionality(self, solution, factor):
        mask = np.random.rand(self.dim) < factor
        reduced_solution = solution.copy()
        reduced_solution[mask] = self.best_solution[mask] if self.best_solution is not None else 0
        return reduced_solution

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
            
            # Dynamic reduction factor based on fitness progress
            reduction_factor = 1 - (evaluations / self.budget) * (self.best_fitness / (self.best_fitness + np.mean(fitness) + 1e-9))

            # Compute mean point for adaptive opposition
            mean_point = np.mean(self.whales, axis=0)

            # Update whales based on the best solution and selective oppositional learning
            for i in range(self.population_size):
                if evaluations >= self.budget:
                    break
                if np.random.rand() < 0.5:
                    # Update using best solution
                    D = np.abs(np.random.rand(self.dim) * self.best_solution - self.whales[i])
                    A = 2 * np.random.rand(self.dim) - 1  # Coefficient for exploration/exploitation balance
                    self.whales[i] = self.best_solution - A * D
                else:
                    # Update using selective oppositional solution
                    if np.random.rand() < 0.7:  # More likely to use mean-based opposition
                        opp_solution = self.oppositional_solution(self.whales[i], mean_point)
                    else:
                        opp_solution = self.oppositional_solution(self.whales[i], (self.lower_bound + self.upper_bound) / 2)
                    D = np.abs(np.random.rand(self.dim) * opp_solution - self.whales[i])
                    A = 2 * np.random.rand(self.dim) - 1
                    self.whales[i] = opp_solution - A * D

                # Apply dimensionality reduction
                self.whales[i] = self.reduce_dimensionality(self.whales[i], reduction_factor)
                
                # Ensure search space boundaries
                self.whales[i] = np.clip(self.whales[i], self.lower_bound, self.upper_bound)

        return self.best_solution, self.best_fitness