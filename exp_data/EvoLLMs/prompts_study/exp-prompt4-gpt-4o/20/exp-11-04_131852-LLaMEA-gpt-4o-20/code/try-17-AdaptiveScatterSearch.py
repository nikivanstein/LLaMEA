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
        F = np.random.uniform(0.4, 0.9)  # Broader range for F to enhance exploration
        mutant_vector = np.clip(a + F * (b - c), self.lower_bound, self.upper_bound)
        return mutant_vector

    def crossover(self, target, mutant):
        cross_prob = 0.9 - 0.3 * (self.num_evaluations / self.budget)
        return np.array([mutant[i] if np.random.rand() < cross_prob else target[i] for i in range(self.dim)])

    def local_search(self, solution, func):
        step_size = 0.2 / (1 + np.exp(-0.1 * self.num_evaluations))
        new_solution = solution + np.random.uniform(-step_size, step_size, self.dim)
        new_solution = np.clip(new_solution, self.lower_bound, self.upper_bound)
        if func(new_solution) < func(solution):
            return new_solution
        return solution

    def stochastic_ranking(self, population, fitness):
        probabilities = np.random.rand(len(fitness))
        ranked = sorted(range(len(fitness)), key=lambda i: (fitness[i], probabilities[i]))
        return [population[i] for i in ranked]

    def __call__(self, func):
        pop_size = 10 + int(10 * (1 - (self.num_evaluations / self.budget)))
        population = self.initialize_population(pop_size)
        fitness = self.evaluate_population(population, func)

        while self.num_evaluations < self.budget:
            population = self.stochastic_ranking(population, fitness)
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

        best_idx = np.argmin(fitness)
        return population[best_idx]