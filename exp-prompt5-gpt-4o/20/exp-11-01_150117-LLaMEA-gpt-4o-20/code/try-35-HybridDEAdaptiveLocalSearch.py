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

            # Adaptive local search
            best_idx = np.argmin(fitness)
            best_sol = pop[best_idx]

            for j in range(self.dim):
                local_sol = best_sol.copy()
                # Adaptive step size
                step_size = np.random.normal(0, 0.1 * (1 - self.current_budget/self.budget))
                local_sol[j] += step_size * np.sign(np.random.rand() - 0.5)  # Directional adjustment
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