import numpy as np

class EnhancedAdaptiveGeneticAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 50
        self.mutation_rate = 0.1
        self.crossover_rate = 0.7
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
        selected_indices = np.random.choice(elite_indices, self.population_size // 2 - elite_size, replace=True)
        return np.concatenate((population[selected_indices], population[elite_indices]))

    def crossover(self, parent1, parent2):
        if np.random.rand() < self.crossover_rate:
            alpha = np.random.rand(self.dim)
            return np.clip(alpha * parent1 + (1 - alpha) * parent2, self.lower_bound, self.upper_bound)
        return parent1 if np.random.rand() < 0.5 else parent2

    def mutate(self, individual):
        mutation_vector = np.random.normal(0, self.mutation_rate, self.dim)
        return np.clip(individual + mutation_vector, self.lower_bound, self.upper_bound)

    def run_generation(self, population, func):
        fitness = self.evaluate_population(population, func)
        parents = self.select_parents(population, fitness)
        new_population = [population[np.argmin(fitness)]]

        for _ in range((self.population_size - 1) // 2):
            parent1, parent2 = parents[np.random.choice(len(parents), 2, replace=False)]
            child = self.crossover(parent1, parent2)
            child = self.mutate(child)
            new_population.append(child)
            self.current_evals += 1
            if self.current_evals >= self.budget:
                break

        return np.array(new_population)

    def update_parameters(self, population, fitness):
        diversity = np.std(population, axis=0).mean()
        fitness_variance = np.var(fitness)
        self.mutation_rate = max(0.01, 0.2 / (self.dim * (1 + diversity)))
        self.crossover_rate = min(1.0, 0.6 + fitness_variance / 100.0)
        self.population_size = int(max(20, min(100, self.population_size * (1 + 0.1 * (diversity - 0.1)))))

    def __call__(self, func):
        population = self.initialize_population()

        while self.current_evals < self.budget:
            fitness = self.evaluate_population(population, func)
            population = self.run_generation(population, func)
            self.update_parameters(population, fitness)

        fitness = self.evaluate_population(population, func)
        best_index = np.argmin(fitness)
        return population[best_index], fitness[best_index]