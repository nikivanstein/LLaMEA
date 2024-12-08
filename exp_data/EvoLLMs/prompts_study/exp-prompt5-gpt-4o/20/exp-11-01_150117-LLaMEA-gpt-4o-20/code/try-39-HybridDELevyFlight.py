import numpy as np

class HybridDELevyFlight:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.bounds = (-5.0, 5.0)
        self.population_size = 10 * dim
        self.F = 0.5  # Differential weight
        self.CR = 0.9  # Crossover probability
        self.current_budget = 0

    def levy_flight(self, Lmbda):
        sigma = (np.math.gamma(1 + Lmbda) * np.sin(np.pi * Lmbda / 2) /
                 (np.math.gamma((1 + Lmbda) / 2) * Lmbda * 2**((Lmbda - 1) / 2)))**(1 / Lmbda)
        u = np.random.normal(0, sigma, size=self.dim)
        v = np.random.normal(0, 1, size=self.dim)
        step = u / abs(v)**(1 / Lmbda)
        return step

    def __call__(self, func):
        # Initialize the population
        pop = np.random.uniform(self.bounds[0], self.bounds[1], (self.population_size, self.dim))
        fitness = np.apply_along_axis(func, 1, pop)
        self.current_budget += self.population_size

        # Main loop
        while self.current_budget < self.budget:
            for i in range(self.population_size):
                # Mutation and recombination
                indices = np.random.choice(self.population_size, 3, replace=False)
                x0, x1, x2 = pop[indices]
                F_dynamic = self.F + (0.5/self.current_budget)
                mutant = x0 + F_dynamic * (x1 - x2)  # Dynamic scaling
                mutant = np.clip(mutant, self.bounds[0], self.bounds[1])

                crossover = np.random.rand(self.dim) < self.CR
                trial = np.where(crossover, mutant, pop[i])

                # Selection
                trial_fitness = func(trial)
                self.current_budget += 1

                if trial_fitness < fitness[i]:
                    pop[i] = trial
                    fitness[i] = trial_fitness

                if self.current_budget >= self.budget:
                    break

            # Adaptive local search with Lévy flight
            best_idx = np.argmin(fitness)
            best_sol = pop[best_idx]
            levy_step = self.levy_flight(1.5)

            local_sol = best_sol + levy_step * (1 - self.current_budget/self.budget)  # Using Lévy flight
            local_sol = np.clip(local_sol, self.bounds[0], self.bounds[1])

            local_fitness = func(local_sol)
            self.current_budget += 1

            if local_fitness < fitness[best_idx]:
                pop[best_idx] = local_sol
                fitness[best_idx] = local_fitness

            if self.current_budget >= self.budget:
                break

        best_idx = np.argmin(fitness)
        return pop[best_idx], fitness[best_idx]