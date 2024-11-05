import numpy as np

class HybridDEAdaptiveLocalSearch:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.bounds = (-5.0, 5.0)
        self.population_size = 10 * dim
        self.F = 0.5  # Differential weight
        self.CR_initial = 0.9  # Initial crossover probability
        self.current_budget = 0

    def __call__(self, func):
        # Initialize the population
        pop = np.random.uniform(self.bounds[0], self.bounds[1], (self.population_size, self.dim))
        fitness = np.apply_along_axis(func, 1, pop)
        self.current_budget += self.population_size

        # Main loop
        while self.current_budget < self.budget:
            for i in range(self.population_size):
                # Dynamic crossover probability
                CR_dynamic = self.CR_initial * (1 - self.current_budget/self.budget)
                # Mutation and recombination
                indices = np.random.choice(self.population_size, 3, replace=False)
                x0, x1, x2 = pop[indices]
                F_dynamic = self.F + (0.5/self.current_budget)
                mutant = x0 + F_dynamic * (x1 - x2)  # Dynamic scaling
                mutant = np.clip(mutant, self.bounds[0], self.bounds[1])

                crossover = np.random.rand(self.dim) < CR_dynamic
                trial = np.where(crossover, mutant, pop[i])

                # Multi-trial selection
                trial_fitness = func(trial)
                self.current_budget += 1

                if trial_fitness < fitness[i]:
                    pop[i] = trial
                    fitness[i] = trial_fitness
                else:
                    trial_2 = x0 + F_dynamic * (pop[np.random.randint(self.population_size)] - x2)
                    trial_2 = np.clip(trial_2, self.bounds[0], self.bounds[1])
                    trial_2_fitness = func(trial_2)
                    self.current_budget += 1
                    if trial_2_fitness < fitness[i]:
                        pop[i] = trial_2
                        fitness[i] = trial_2_fitness

                if self.current_budget >= self.budget:
                    break

            # Adaptive local search
            best_idx = np.argmin(fitness)
            best_sol = pop[best_idx]

            for j in range(self.dim):
                local_sol = best_sol.copy()
                local_sol[j] += np.random.normal(0, 0.1 * (1 - self.current_budget/self.budget))  # Adaptive step size
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