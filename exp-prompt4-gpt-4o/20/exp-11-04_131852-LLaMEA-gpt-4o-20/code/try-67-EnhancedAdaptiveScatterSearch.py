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

    def differential_mutation(self, target_idx, population, fitness):
        idxs = [idx for idx in range(len(population)) if idx != target_idx]
        a, b, c = population[np.random.choice(idxs, 3, replace=False)]
        best_idx = np.argmin(fitness)
        best = population[best_idx]
        F = np.random.uniform(0.4, 0.8)  # Adjusted F range for better stability
        mutant_vector = np.clip(best + F * (a - b) + F * (b - c), self.lower_bound, self.upper_bound)
        return mutant_vector

    def crossover(self, target, mutant):
        cross_prob = 0.9 - 0.6 * (self.num_evaluations / self.budget)  # Enhanced dynamic crossover probability
        return np.array([mutant[i] if np.random.rand() < cross_prob else target[i] for i in range(self.dim)])

    def levy_flight(self, L=1.5):
        u = np.random.normal(0, 1, self.dim) * (1 / np.power(np.abs(np.random.normal(0, 1, self.dim)), 1/L))
        return u

    def local_search(self, solution, func):
        step_size = 0.15 / (1 + np.exp(-0.1 * self.num_evaluations))  # Tighter step size for refined search
        new_solution = solution + self.levy_flight() * step_size
        new_solution = np.clip(new_solution, self.lower_bound, self.upper_bound)
        if func(new_solution) < func(solution):
            return new_solution
        return solution

    def __call__(self, func):
        best_solution = None
        best_fitness = np.inf
        while self.num_evaluations < self.budget:
            pop_size = 12 + int(8 * (1 - (self.num_evaluations / self.budget)))  # Dynamic pop size
            population = self.initialize_population(pop_size)
            fitness = self.evaluate_population(population, func)

            for i in range(len(population)):
                mutant = self.differential_mutation(i, population, fitness)
                trial = self.crossover(population[i], mutant)
                trial = self.local_search(trial, func)

                trial_fitness = func(trial)
                self.num_evaluations += 1
                if trial_fitness < fitness[i]:
                    population[i] = trial
                    fitness[i] = trial_fitness
                    if trial_fitness < best_fitness:
                        best_solution = trial
                        best_fitness = trial_fitness

                if self.num_evaluations >= self.budget:
                    break

        return best_solution