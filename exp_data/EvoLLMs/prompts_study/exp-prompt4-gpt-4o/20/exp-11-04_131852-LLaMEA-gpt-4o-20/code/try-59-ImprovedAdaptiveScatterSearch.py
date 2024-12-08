import numpy as np

class ImprovedAdaptiveScatterSearch:
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
        F = np.random.uniform(0.5, 0.9)  # Increased F range for exploration
        mutant_vector = np.clip(best + F * (a - b), self.lower_bound, self.upper_bound)  # Simplified strategy
        return mutant_vector

    def crossover(self, target, mutant):
        cross_prob = 0.9 - 0.6 * (self.num_evaluations / self.budget)  # More dynamic crossover probability
        return np.array([mutant[i] if np.random.rand() < cross_prob else target[i] for i in range(self.dim)])
    
    def levy_flight(self, L=1.5):
        u = np.random.normal(0, 1, self.dim) * (1 / np.power(np.abs(np.random.normal(0, 1, self.dim)), 1/L))
        return u
    
    def local_search(self, solution, func, fitness_value):
        step_size = 0.1 + 0.9 * (1 - fitness_value / np.max(fitness_value))
        new_solution = solution + self.levy_flight() * step_size
        new_solution = np.clip(new_solution, self.lower_bound, self.upper_bound)
        if func(new_solution) < func(solution):
            return new_solution
        return solution

    def __call__(self, func):
        pop_size = 15 + int(5 * (1 - (self.num_evaluations / self.budget)))  # Adjusted dynamic resizing
        population = self.initialize_population(pop_size)
        fitness = self.evaluate_population(population, func)

        while self.num_evaluations < self.budget:
            for i in range(len(population)):
                mutant = self.differential_mutation(i, population, fitness)
                trial = self.crossover(population[i], mutant)
                trial = self.local_search(trial, func, fitness)

                trial_fitness = func(trial)
                self.num_evaluations += 1
                if trial_fitness < fitness[i]:
                    population[i] = trial
                    fitness[i] = trial_fitness

                if self.num_evaluations >= self.budget:
                    break

        best_idx = np.argmin(fitness)
        return population[best_idx]