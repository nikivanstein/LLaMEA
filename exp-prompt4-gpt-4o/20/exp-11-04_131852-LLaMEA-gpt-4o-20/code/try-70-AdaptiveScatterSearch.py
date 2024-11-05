import numpy as np

class AdaptiveScatterSearch:
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
        F = np.random.uniform(0.4, 0.8)  # Slightly modified F range
        mutant_vector = np.clip(best + F * (a - c), self.lower_bound, self.upper_bound)  # Revised strategy
        return mutant_vector

    def crossover(self, target, mutant):
        cross_prob = 0.75 - 0.45 * (self.num_evaluations / self.budget)  # Adjusted crossover probability
        return np.array([mutant[i] if np.random.rand() < cross_prob else target[i] for i in range(self.dim)])

    def levy_flight(self, L=1.8):  # Enhanced Levy flight
        u = np.random.normal(0, 1, self.dim) * (1 / np.power(np.abs(np.random.normal(0, 1, self.dim)), 1/L))
        return u

    def local_search(self, solution, func):
        step_size = 0.3 / (1 + np.exp(-0.1 * self.num_evaluations))  # Adjusted step size
        new_solution = solution + self.levy_flight() * step_size
        new_solution = np.clip(new_solution, self.lower_bound, self.upper_bound)
        if func(new_solution) < func(solution):
            return new_solution
        return solution

    def centralized_selection(self, population, fitness):
        central_point = np.mean(population, axis=0)
        distances = np.linalg.norm(population - central_point, axis=1)
        best_idx = np.argmin(fitness + distances * 0.1)  # Promote central convergence
        return population[best_idx], fitness[best_idx]

    def __call__(self, func):
        pop_size = 10 + int(10 * (1 - (self.num_evaluations / self.budget)))
        population = self.initialize_population(pop_size)
        fitness = self.evaluate_population(population, func)

        while self.num_evaluations < self.budget:
            for i in range(len(population)):
                mutant = self.differential_mutation(i, population, fitness)
                trial = self.crossover(population[i], mutant)
                trial = self.local_search(trial, func)

                trial_fitness = func(trial)
                self.num_evaluations += 1
                if trial_fitness < fitness[i]:
                    population[i] = trial
                    fitness[i] = trial_fitness

                if self.num_evaluations >= self.budget:
                    break

        best_solution, best_fitness = self.centralized_selection(population, fitness)
        return best_solution