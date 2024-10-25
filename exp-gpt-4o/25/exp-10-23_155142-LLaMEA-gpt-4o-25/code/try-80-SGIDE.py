import numpy as np

class SGIDE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.pop_size = 10 * dim
        self.F = 0.5  # Differential weight
        self.Cr = 0.9  # Crossover probability
        self.gradient_memory = []
        self.best_solution = None
        self.elite_percentage = 0.1  # Preserve top 10% elites

    def initialize_population(self):
        return self.lower_bound + (self.upper_bound - self.lower_bound) * np.random.rand(self.pop_size, self.dim)

    def approximate_gradient(self, func, sol):
        grad = np.zeros(self.dim)
        epsilon = 1e-8
        for j in range(self.dim):
            sol_forward = np.copy(sol)
            sol_backward = np.copy(sol)
            sol_forward[j] += epsilon
            sol_backward[j] -= epsilon
            grad[j] = (func(sol_forward) - func(sol_backward)) / (2 * epsilon)
        return grad

    def __call__(self, func):
        population = self.initialize_population()
        fitness = np.array([func(ind) for ind in population])
        eval_count = self.pop_size

        while eval_count < self.budget:
            new_population = np.copy(population)
            elite_count = int(self.pop_size * self.elite_percentage)
            elite_indices = np.argsort(fitness)[:elite_count]
            new_population[elite_indices] = population[elite_indices]

            for i in range(self.pop_size):
                if i in elite_indices:
                    continue

                x_best = population[np.argmin(fitness)] if self.best_solution is None else self.best_solution
                indices = np.random.choice([j for j in range(self.pop_size) if j != i], 3, replace=False)
                x1, x2, x3 = population[indices]
                
                gradient = self.approximate_gradient(func, population[i])
                self.F = 0.5 + 0.5 * np.tanh(np.linalg.norm(gradient))
                mutant = np.clip(x_best + self.F * (x1 - x2) + 0.1 * gradient, self.lower_bound, self.upper_bound)

                crossover_mask = np.random.rand(self.dim) < self.Cr
                trial = np.where(crossover_mask, mutant, population[i])

                trial_fitness = func(trial)
                eval_count += 1
                if trial_fitness < fitness[i]:
                    new_population[i] = trial
                    fitness[i] = trial_fitness

                    if self.best_solution is None or trial_fitness < func(self.best_solution):
                        self.best_solution = trial

                if eval_count >= self.budget:
                    break

            population = new_population

        best_idx = np.argmin(fitness)
        return population[best_idx], fitness[best_idx]