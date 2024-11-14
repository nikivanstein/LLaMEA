import numpy as np

class ADE_LS:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lb = -5.0
        self.ub = 5.0
        self.population_size = 5 * dim
        self.scaling_factor = 0.8
        self.crossover_rate = 0.7
        self.current_evaluations = 0

    def __call__(self, func):
        # Initialize population
        population = np.random.uniform(self.lb, self.ub, (self.population_size, self.dim))
        fitness = np.apply_along_axis(func, 1, population)
        self.current_evaluations += self.population_size

        while self.current_evaluations < self.budget:
            for i in range(self.population_size):
                # Mutation
                indices = np.random.choice(self.population_size, 3, replace=False)
                while i in indices:
                    indices = np.random.choice(self.population_size, 3, replace=False)
                x1, x2, x3 = population[indices]
                mutant = np.clip(x1 + self.scaling_factor * (x2 - x3), self.lb, self.ub)

                # Crossover
                trial = np.where(np.random.rand(self.dim) < self.crossover_rate, mutant, population[i])

                # Local Search (e.g., simple hill climbing)
                local_step = np.random.uniform(-0.1, 0.1, self.dim)
                trial_ls = np.clip(trial + local_step, self.lb, self.ub)
                trial_ls_fitness = func(trial_ls)
                self.current_evaluations += 1

                if trial_ls_fitness < fitness[i]:
                    trial = trial_ls
                    trial_fitness = trial_ls_fitness
                else:
                    trial_fitness = func(trial)
                    self.current_evaluations += 1

                # Selection
                if trial_fitness < fitness[i]:
                    population[i] = trial
                    fitness[i] = trial_fitness

                # Check budget
                if self.current_evaluations >= self.budget:
                    break
            
        # Return the best solution found
        best_index = np.argmin(fitness)
        best_solution = population[best_index]
        best_fitness = fitness[best_index]
        return best_solution, best_fitness