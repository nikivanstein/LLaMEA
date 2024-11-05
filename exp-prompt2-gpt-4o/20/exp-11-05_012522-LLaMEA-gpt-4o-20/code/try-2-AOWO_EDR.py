import numpy as np

class AOWO_EDR:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.initial_population_size = min(30, self.budget // 5)
        self.population_size = self.initial_population_size
        self.whales = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        self.best_solution = None
        self.best_fitness = float('inf')
    
    def oppositional_solution(self, solution):
        return self.lower_bound + self.upper_bound - solution

    def reduce_dimensionality(self, solution, factor):
        mask = np.random.rand(self.dim) < factor
        reduced_solution = solution.copy()
        reduced_solution[mask] = self.best_solution[mask] if self.best_solution is not None else 0
        return reduced_solution

    def adaptive_mutation(self, solution, evaluations):
        mutation_rate = (1 - evaluations / self.budget) * 0.1
        mutation_vector = np.random.uniform(-mutation_rate, mutation_rate, self.dim)
        return np.clip(solution + mutation_vector, self.lower_bound, self.upper_bound)

    def __call__(self, func):
        evaluations = 0

        while evaluations < self.budget:
            # Dynamically adjust population size
            self.population_size = self.initial_population_size + int((self.budget - evaluations) / (2 * self.dim))
            self.whales = self.whales[:self.population_size]

            # Calculate fitness for current population
            fitness = np.array([func(whale) for whale in self.whales])
            evaluations += self.population_size

            # Update best solution found
            min_fitness_idx = np.argmin(fitness)
            if fitness[min_fitness_idx] < self.best_fitness:
                self.best_fitness = fitness[min_fitness_idx]
                self.best_solution = self.whales[min_fitness_idx].copy()
            
            # Dimensionality reduction factor adapts over iterations
            reduction_factor = 1 - (evaluations / self.budget)

            # Update whales based on the best solution and oppositional learning
            for i in range(self.population_size):
                if evaluations >= self.budget:
                    break
                if np.random.rand() < 0.5:
                    # Update using best solution
                    D = np.abs(np.random.rand(self.dim) * self.best_solution - self.whales[i])
                    A = 2 * np.random.rand(self.dim) - 1
                    self.whales[i] = self.best_solution - A * D
                else:
                    # Update using oppositional solution
                    opp_solution = self.oppositional_solution(self.whales[i])
                    D = np.abs(np.random.rand(self.dim) * opp_solution - self.whales[i])
                    A = 2 * np.random.rand(self.dim) - 1
                    self.whales[i] = opp_solution - A * D

                # Apply dimensionality reduction
                self.whales[i] = self.reduce_dimensionality(self.whales[i], reduction_factor)

                # Apply adaptive mutation
                self.whales[i] = self.adaptive_mutation(self.whales[i], evaluations)

                # Ensure search space boundaries
                self.whales[i] = np.clip(self.whales[i], self.lower_bound, self.upper_bound)

        return self.best_solution, self.best_fitness