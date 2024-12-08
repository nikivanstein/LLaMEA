import numpy as np

class EnhancedAdaptiveScatterSearch:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.num_evaluations = 0
        self.lower_bound = -5.0
        self.upper_bound = 5.0

    def initialize_population(self, size):
        return np.random.uniform(self.lower_bound, self.upper_bound, (size, self.dim))

    def evaluate_population(self, population, func):
        fitness = np.apply_along_axis(func, 1, population)
        self.num_evaluations += len(fitness)
        return fitness

    def adaptive_mutation(self, target_idx, population, fitness):
        idxs = [idx for idx in range(len(population)) if idx != target_idx]
        a, b, c = population[np.random.choice(idxs, 3, replace=False)]
        best_idx = np.argmin(fitness)
        best = population[best_idx]
        F = 0.5 + 0.5 * (self.num_evaluations / self.budget)  # Adaptive scaling factor
        mutant_vector = np.clip(best + F * (a - b), self.lower_bound, self.upper_bound)
        return mutant_vector

    def crossover(self, target, mutant):
        cross_prob = 0.9 - 0.7 * (self.num_evaluations / self.budget)  # Adjusted crossover probability for more exploration at the start
        return np.array([mutant[i] if np.random.rand() < cross_prob else target[i] for i in range(self.dim)])

    def local_search(self, solution, func):
        step_size = 0.1 / (1 + np.exp(-0.05 * self.num_evaluations))  # Reduced step size for finer local search
        new_solution = solution + self.levy_flight() * step_size
        new_solution = np.clip(new_solution, self.lower_bound, self.upper_bound)
        if func(new_solution) < func(solution):
            return new_solution
        return solution

    def levy_flight(self, L=1.5):
        u = np.random.normal(0, 1, self.dim) * (1 / np.power(np.abs(np.random.normal(0, 1, self.dim)), 1/L))
        return u

    def __call__(self, func):
        pop_size = 10 + int(10 * (1 - (self.num_evaluations / self.budget)))
        population = self.initialize_population(pop_size)
        fitness = self.evaluate_population(population, func)

        while self.num_evaluations < self.budget:
            for i in range(len(population)):
                mutant = self.adaptive_mutation(i, population, fitness)
                trial = self.crossover(population[i], mutant)
                trial = self.local_search(trial, func)

                trial_fitness = func(trial)
                self.num_evaluations += 1
                if trial_fitness < fitness[i]:
                    population[i] = trial
                    fitness[i] = trial_fitness

                if self.num_evaluations >= self.budget:
                    break

        best_idx = np.argmin(fitness)
        return population[best_idx]