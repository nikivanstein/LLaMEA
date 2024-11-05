import numpy as np

class HybridDEAdaptiveLocalSearch:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.bounds = (-5.0, 5.0)
        self.population_size = 10 * dim
        self.F = 0.5  # Differential weight
        self.CR = 0.9  # Crossover probability
        self.current_budget = 0

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

            # Self-adaptive control for F and CR
            self.F = np.random.uniform(0.4, 0.9)
            self.CR = np.random.uniform(0.5, 1.0)

            # Neighborhood-based local search
            best_idx = np.argmin(fitness)
            best_sol = pop[best_idx]

            for j in range(self.dim):
                local_sol = best_sol.copy()
                for _ in range(3):  # Neighborhood exploration
                    neighbor = local_sol + np.random.uniform(-0.1, 0.1, self.dim)
                    neighbor = np.clip(neighbor, self.bounds[0], self.bounds[1])
                    neighbor_fitness = func(neighbor)
                    self.current_budget += 1

                    if neighbor_fitness < fitness[best_idx]:
                        pop[best_idx] = neighbor
                        fitness[best_idx] = neighbor_fitness

                    if self.current_budget >= self.budget:
                        break

        best_idx = np.argmin(fitness)
        return pop[best_idx], fitness[best_idx]