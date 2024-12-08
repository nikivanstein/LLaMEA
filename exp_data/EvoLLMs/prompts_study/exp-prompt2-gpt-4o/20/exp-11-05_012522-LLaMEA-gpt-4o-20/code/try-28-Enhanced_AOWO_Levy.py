import numpy as np

class Enhanced_AOWO_Levy:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = min(30, self.budget // 5)
        self.whales = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        self.best_solution = None
        self.best_fitness = float('inf')
        self.dynamic_scale = 0.5  # Initial dynamic scaling factor

    def oppositional_solution(self, solution):
        return self.lower_bound + self.upper_bound - solution

    def levy_flight(self, dim, beta=1.5):
        sigma_u = (np.math.gamma(1 + beta) * np.sin(np.pi * beta / 2) / 
                   (np.math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (1 / beta)
        u = np.random.normal(0, sigma_u, size=dim)
        v = np.random.normal(0, 1, size=dim)
        step = u / np.abs(v) ** (1 / beta)
        return step

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
            
            # Dimensionality reduction factor adapts over iterations
            reduction_factor = 1 - (evaluations / self.budget)

            # Update whales with dynamic scaling and oppositional learning
            for i in range(self.population_size):
                if evaluations >= self.budget:
                    break
                # Adaptive dynamic scaling
                self.dynamic_scale = 0.5 * (1 + np.sin(np.pi * evaluations / self.budget))

                if np.random.rand() < 0.5:
                    # Update using best solution with Levy flight
                    levy_step = self.levy_flight(self.dim) * self.dynamic_scale
                    self.whales[i] = self.best_solution + levy_step
                else:
                    # Update using oppositional solution
                    opp_solution = self.oppositional_solution(self.whales[i])
                    D = np.abs(np.random.rand(self.dim) * opp_solution - self.whales[i])
                    A = 2 * np.random.rand(self.dim) - 1
                    self.whales[i] = opp_solution - A * D * self.dynamic_scale

                # Apply dimensionality reduction
                self.whales[i] = self.whales[i] * reduction_factor
                
                # Ensure search space boundaries
                self.whales[i] = np.clip(self.whales[i], self.lower_bound, self.upper_bound)

        return self.best_solution, self.best_fitness