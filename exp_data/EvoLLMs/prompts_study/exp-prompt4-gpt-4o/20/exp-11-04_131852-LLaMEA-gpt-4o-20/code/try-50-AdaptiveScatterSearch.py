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

    def hybrid_mutation(self, target_idx, population, fitness):
        idxs = [idx for idx in range(len(population)) if idx != target_idx]
        a, b, c, d = population[np.random.choice(idxs, 4, replace=False)]
        F1 = np.random.uniform(0.4, 0.8)
        F2 = np.random.uniform(0.1, 0.4)
        mutant_vector = np.clip(a + F1 * (b - c) + F2 * (d - b), self.lower_bound, self.upper_bound)
        return mutant_vector

    def crossover(self, target, mutant):
        cross_prob = 0.9 - 0.3 * (self.num_evaluations / self.budget)
        return np.array([mutant[i] if np.random.rand() < cross_prob else target[i] for i in range(self.dim)])

    def levy_flight(self, L=1.5):
        u = np.random.normal(0, 1, self.dim) * (1 / np.power(np.abs(np.random.normal(0, 1, self.dim)), 1/L))
        return u

    def local_search(self, solution, func):
        step_size = 0.2 / (1 + np.exp(-0.1 * self.num_evaluations))
        new_solution = solution + self.levy_flight() * step_size  # Implementing Levy flights
        new_solution = np.clip(new_solution, self.lower_bound, self.upper_bound)
        if func(new_solution) < func(solution):
            return new_solution
        return solution

    def __call__(self, func):
        initial_pop_size = 15
        population = self.initialize_population(initial_pop_size)
        fitness = self.evaluate_population(population, func)

        while self.num_evaluations < self.budget:
            for i in range(len(population)):
                mutant = self.hybrid_mutation(i, population, fitness)
                trial = self.crossover(population[i], mutant)
                trial = self.local_search(trial, func)

                trial_fitness = func(trial)
                self.num_evaluations += 1
                if trial_fitness < fitness[i]:
                    population[i] = trial
                    fitness[i] = trial_fitness

                if self.num_evaluations >= self.budget:
                    break

            if self.num_evaluations < self.budget * 0.5:
                new_pop_size = len(population) + int(5 * (self.num_evaluations / self.budget))
                population = np.vstack((population, self.initialize_population(new_pop_size - len(population))))
                fitness = np.hstack((fitness, self.evaluate_population(population[len(fitness):], func)))

        best_idx = np.argmin(fitness)
        return population[best_idx]