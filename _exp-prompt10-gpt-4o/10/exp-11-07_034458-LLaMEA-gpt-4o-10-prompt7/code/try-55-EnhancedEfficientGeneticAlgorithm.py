import numpy as np
from concurrent.futures import ThreadPoolExecutor

class EnhancedEfficientGeneticAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 50
        self.mutation_rate = 0.1
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.current_evals = 0

    def initialize_population(self):
        return np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))

    def evaluate_population(self, population, func):
        return np.apply_along_axis(func, 1, population)

    def select_parents(self, population, fitness):
        elite_size = self.population_size // 10
        elite_indices = np.argsort(fitness)[:elite_size]
        weights = 1.0 / (np.argsort(fitness) + 1)
        selected_indices = np.random.choice(len(fitness), self.population_size // 2 - elite_size, p=weights/weights.sum())
        return np.concatenate((population[selected_indices], population[elite_indices]))

    def crossover(self, parent1, parent2):
        return (parent1 + parent2) / 2

    def mutate_individual(self, individual):
        mutation_vector = np.random.normal(0, self.mutation_rate, self.dim)
        return np.clip(individual + mutation_vector, self.lower_bound, self.upper_bound)

    def run_generation(self, population, func):
        fitness = self.evaluate_population(population, func)
        parents = self.select_parents(population, fitness)
        new_population = []

        with ThreadPoolExecutor(max_workers=4) as executor:
            for _ in range(self.population_size // 2):
                parent1, parent2 = parents[np.random.choice(len(parents), 2, replace=False)]
                child = self.crossover(parent1, parent2)
                future = executor.submit(self.mutate_individual, child)
                new_population.append(future.result())
                self.current_evals += 1
                if self.current_evals >= self.budget:
                    break

        return np.array(new_population)

    def update_parameters(self, population):
        diversity = np.std(population, axis=0).mean()
        self.mutation_rate = max(0.01, 0.2 / (self.dim * (1 + diversity)))
        self.population_size = int(max(20, min(100, self.population_size * (1 + 0.1 * (diversity - 0.1)))))

    def __call__(self, func):
        population = self.initialize_population()

        while self.current_evals < self.budget:
            population = self.run_generation(population, func)
            self.update_parameters(population)

        fitness = self.evaluate_population(population, func)
        best_index = np.argmin(fitness)
        return population[best_index], fitness[best_index]