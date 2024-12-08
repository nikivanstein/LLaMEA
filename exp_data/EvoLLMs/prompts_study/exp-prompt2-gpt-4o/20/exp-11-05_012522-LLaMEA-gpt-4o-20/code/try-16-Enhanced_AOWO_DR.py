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
        self.elite_rate = 0.2  # New: Rate of elite solutions

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
            # Calculate fitness for current population
            fitness = np.array([func(whale) for whale in self.whales])
            evaluations += self.population_size

            # Update best solution found
            elite_count = max(1, int(self.elite_rate * self.population_size))
            elite_indices = np.argsort(fitness)[:elite_count]
            elite_solutions = self.whales[elite_indices]

            if fitness[elite_indices[0]] < self.best_fitness:
                self.best_fitness = fitness[elite_indices[0]]
                self.best_solution = self.whales[elite_indices[0]].copy()

            # Dimensionality reduction factor adapts over iterations
            reduction_factor = 1 - (evaluations / self.budget)

            # Update whales based on elite solutions and oppositional learning
            for i in range(self.population_size):
                if evaluations >= self.budget:
                    break
                if np.random.rand() < 0.5:
                    # Update using an elite solution
                    elite_whale = elite_solutions[np.random.randint(elite_count)]
                    D = np.abs(np.random.rand(self.dim) * elite_whale - self.whales[i])
                    A = 2 * np.random.rand(self.dim) - 1
                    self.whales[i] = elite_whale - A * D
                else:
                    # Update using oppositional solution
                    opp_solution = self.oppositional_solution(self.whales[i])
                    D = np.abs(np.random.rand(self.dim) * opp_solution - self.whales[i])
                    A = 2 * np.random.rand(self.dim) - 1
                    self.whales[i] = opp_solution - A * D

                # Apply dimensionality reduction
                self.whales[i] = self.reduce_dimensionality(self.whales[i], reduction_factor)

                # Ensure search space boundaries
                self.whales[i] = np.clip(self.whales[i], self.lower_bound, self.upper_bound)

        return self.best_solution, self.best_fitness