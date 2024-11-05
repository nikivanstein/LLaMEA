import numpy as np

class AOWO_DR:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = min(30, self.budget // 5)
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
    
    def crossover(self, target, donor):
        cr = 0.9
        mask = np.random.rand(self.dim) < cr
        return np.where(mask, donor, target)

    def __call__(self, func):
        evaluations = 0
        F = 0.5  # Differential weight

        while evaluations < self.budget:
            fitness = np.array([func(whale) for whale in self.whales])
            evaluations += self.population_size

            min_fitness_idx = np.argmin(fitness)
            if fitness[min_fitness_idx] < self.best_fitness:
                self.best_fitness = fitness[min_fitness_idx]
                self.best_solution = self.whales[min_fitness_idx].copy()
            
            reduction_factor = 1 - (evaluations / self.budget)

            for i in range(self.population_size):
                if evaluations >= self.budget:
                    break

                r1, r2 = np.random.choice(self.population_size, 2, replace=False)
                donor_vector = self.whales[r1] + F * (self.whales[r2] - self.whales[i])
                donor_vector = np.clip(donor_vector, self.lower_bound, self.upper_bound)
                trial_vector = self.crossover(self.whales[i], donor_vector)

                if np.random.rand() < 0.5:
                    D = np.abs(np.random.rand(self.dim) * self.best_solution - trial_vector)
                    A = 2 * np.random.rand(self.dim) - 1
                    self.whales[i] = trial_vector - A * D
                else:
                    opp_solution = self.oppositional_solution(trial_vector)
                    D = np.abs(np.random.rand(self.dim) * opp_solution - trial_vector)
                    A = 2 * np.random.rand(self.dim) - 1
                    self.whales[i] = opp_solution - A * D

                self.whales[i] = self.reduce_dimensionality(self.whales[i], reduction_factor)
                self.whales[i] = np.clip(self.whales[i], self.lower_bound, self.upper_bound)

        return self.best_solution, self.best_fitness