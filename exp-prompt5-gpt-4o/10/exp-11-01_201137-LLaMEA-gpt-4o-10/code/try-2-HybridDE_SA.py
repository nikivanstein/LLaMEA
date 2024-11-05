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

    def __call__(self, func):
        population = np.random.uniform(self.lower_bound, self.upper_bound, 
                                       (self.population_size, self.dim))
        fitness = np.apply_along_axis(func, 1, population)
        best_idx = np.argmin(fitness)
        best_solution = population[best_idx]
        best_fitness = fitness[best_idx]
        T = self.T0

        evals = self.population_size

        for _ in range(self.iterations):
            for i in range(self.population_size):
                # Differential Evolution Mutation
                indices = np.random.choice(self.population_size, 3, replace=False)
                if best_idx in indices:
                    indices = np.random.choice(self.population_size, 3, replace=False)
                x1, x2, x3 = population[indices]
                mutant = x1 + self.F * (x2 - x3)
                mutant = np.clip(mutant, self.lower_bound, self.upper_bound)
                
                # Adaptive Crossover
                self.CR = 0.9 - (0.8 * (evals / self.budget))
                crossover_mask = np.random.rand(self.dim) < self.CR
                if not np.any(crossover_mask):
                    crossover_mask[np.random.randint(0, self.dim)] = True
                trial = np.where(crossover_mask, mutant, population[i])
                
                # Evaluate trial
                trial_fitness = func(trial)
                evals += 1
                
                # Selection
                if trial_fitness < fitness[i]:
                    population[i] = trial
                    fitness[i] = trial_fitness
                    if trial_fitness < best_fitness:
                        best_solution = trial
                        best_fitness = trial_fitness

            # Simulated Annealing Perturbation
            for i in range(self.population_size):
                perturbed = population[i] + np.random.normal(0, 0.1, self.dim)
                perturbed = np.clip(perturbed, self.lower_bound, self.upper_bound)
                perturbed_fitness = func(perturbed)
                evals += 1
                
                # Metropolis criterion
                if perturbed_fitness < fitness[i] or \
                   np.random.rand() < np.exp((fitness[i] - perturbed_fitness) / T):
                    population[i] = perturbed
                    fitness[i] = perturbed_fitness
                    if perturbed_fitness < best_fitness:
                        best_solution = perturbed
                        best_fitness = perturbed_fitness

            T *= self.alpha  # Cool down the temperature

            if evals >= self.budget:
                break

        return best_solution, best_fitness