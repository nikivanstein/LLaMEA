import numpy as np

class HybridDE_SA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = 10 + 10 * dim
        self.F = 0.8  # Differential Evolution parameter
        self.CR = 0.9  # Crossover rate
        self.T0 = 1000  # Initial temperature for Simulated Annealing
        self.alpha = 0.99  # Cooling rate
        self.iterations = int(budget / self.population_size)
        self.elite_solutions = []
        self.diversity_rate = 0.1  # Reinitialize 10% of the population

    def __call__(self, func):
        population = np.random.uniform(self.lower_bound, self.upper_bound, 
                                       (self.population_size, self.dim))
        fitness = np.apply_along_axis(func, 1, population)
        best_idx = np.argmin(fitness)
        best_solution = population[best_idx]
        best_fitness = fitness[best_idx]
        T = self.T0

        evals = self.population_size

        for it in range(self.iterations):
            self.F = 0.5 + 0.3 * np.sin(np.pi * it / self.iterations)  # Dynamic F
            for i in range(self.population_size):
                indices = np.random.choice(self.population_size, 3, replace=False)
                if best_idx in indices:
                    indices = np.random.choice(self.population_size, 3, replace=False)
                x1, x2, x3 = population[indices]
                mutant = x1 + self.F * (x2 - x3 + np.random.uniform(-0.1, 0.1, self.dim))  # Adapted mutation
                mutant = np.clip(mutant, self.lower_bound, self.upper_bound)
                
                crossover_mask = np.random.rand(self.dim) < self.CR
                if not np.any(crossover_mask):
                    crossover_mask[np.random.randint(0, self.dim)] = True
                trial = np.where(crossover_mask, mutant, population[i])
                
                trial_fitness = func(trial)
                evals += 1
                
                if trial_fitness < fitness[i]:
                    population[i] = trial
                    fitness[i] = trial_fitness
                    if trial_fitness < best_fitness:
                        best_solution = trial
                        best_fitness = trial_fitness
                        self.elite_solutions.append(trial)
            
            for i in range(self.population_size):
                perturbed = population[i] + np.random.normal(0, 0.1, self.dim)
                perturbed = np.clip(perturbed, self.lower_bound, self.upper_bound)
                perturbed_fitness = func(perturbed)
                evals += 1
                
                if perturbed_fitness < fitness[i] or \
                   np.random.rand() < np.exp((fitness[i] - perturbed_fitness) / T):
                    population[i] = perturbed
                    fitness[i] = perturbed_fitness
                    if perturbed_fitness < best_fitness:
                        best_solution = perturbed
                        best_fitness = perturbed_fitness
                        self.elite_solutions.append(perturbed)

            if it % 10 == 0:  # Every 10 iterations
                n_reinit = int(self.population_size * self.diversity_rate)
                reinit_indices = np.random.choice(self.population_size, n_reinit, replace=False)
                population[reinit_indices] = np.random.uniform(self.lower_bound, self.upper_bound, (n_reinit, self.dim))
                fitness[reinit_indices] = np.apply_along_axis(func, 1, population[reinit_indices])
                evals += n_reinit

            T *= self.alpha  # Cool down the temperature
            if len(self.elite_solutions) > 5:  # Selective elitism
                self.elite_solutions = sorted(self.elite_solutions, key=func)[:5]

            if evals >= self.budget:
                break

        return best_solution, best_fitness