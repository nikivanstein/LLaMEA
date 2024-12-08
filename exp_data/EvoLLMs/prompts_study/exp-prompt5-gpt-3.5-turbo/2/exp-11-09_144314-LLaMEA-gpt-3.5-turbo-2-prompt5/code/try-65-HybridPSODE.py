# Hybrid of Particle Swarm Optimization and Differential Evolution algorithm
import numpy as np

class HybridPSODE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 30
        self.max_iterations = budget // self.population_size

    def __call__(self, func):
        def initialize_population():
            return np.random.uniform(-5.0, 5.0, (self.population_size, self.dim))

        def evaluate_population(population):
            return np.array([func(individual) for individual in population])

        def update_population(population, scores):
            best_index = np.argmin(scores)
            best_individual = population[best_index]
            return best_individual
        
        def evolve_population(population):
            new_population = []
            for i in range(self.population_size):
                a, b, c = np.random.choice(population, 3, replace=False)
                mutant = a + 0.5 * (b - c)
                child = np.where(np.random.rand(self.dim) < 0.5, mutant, population[i])
                new_population.append(child)
            return np.array(new_population)

        population = initialize_population()
        scores = evaluate_population(population)
        best_solution = update_population(population, scores)

        for _ in range(self.max_iterations):
            new_population = evolve_population(population)
            new_scores = evaluate_population(new_population)
            population = new_population
            scores = new_scores
            current_best = update_population(population, scores)
            if func(current_best) < func(best_solution):
                best_solution = current_best

        return best_solution