import numpy as np

class HEBBO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = np.linspace(-5.0, 5.0, dim)
        self.func_evaluations = 0
        self.population_size = 100
        self.mutation_rate = 0.01
        self.evolution_strategy = "mutation"

    def __call__(self, func):
        while self.func_evaluations < self.budget:
            func_evaluations += 1
            func_value = func(self.search_space)
            if np.isnan(func_value) or np.isinf(func_value):
                raise ValueError("Invalid function value")
            if func_value < 0 or func_value > 1:
                raise ValueError("Function value must be between 0 and 1")
            self.search_space = np.linspace(-5.0, 5.0, self.dim)
        return func_value

    def initialize_population(self):
        return [self.generate_individual() for _ in range(self.population_size)]

    def generate_individual(self):
        return np.random.uniform(self.search_space, size=self.dim)

    def evolve_population(self, population):
        if self.evolution_strategy == "mutation":
            new_population = []
            for individual in population:
                mutated_individual = individual.copy()
                if np.random.rand() < self.mutation_rate:
                    mutated_individual[np.random.randint(self.dim)] = np.random.uniform(self.search_space[np.random.randint(self.dim)])
                new_population.append(mutated_individual)
            return new_population
        elif self.evolution_strategy == "crossover":
            new_population = []
            for i in range(0, len(population), 2):
                parent1 = population[i]
                parent2 = population[i + 1]
                child = np.concatenate((parent1[:self.dim], parent2[self.dim:]))
                new_population.append(child)
            return new_population
        elif self.evolution_strategy == "selection":
            new_population = []
            for individual in population:
                fitness = self.evaluate_fitness(individual)
                new_population.append(individual)
                if fitness > np.random.rand():
                    new_population.append(individual)
            return new_population

    def evaluate_fitness(self, individual):
        func_value = self.__call__(individual)
        return func_value

    def __call__(self, func):
        while self.func_evaluations < self.budget:
            func_evaluations += 1
            func_value = func(self.search_space)
            if np.isnan(func_value) or np.isinf(func_value):
                raise ValueError("Invalid function value")
            if func_value < 0 or func_value > 1:
                raise ValueError("Function value must be between 0 and 1")
            self.search_space = np.linspace(-5.0, 5.0, self.dim)
        return func_value

# Select the first solution
selected_solution = HEBBO(1000, 10).initialize_population()

# Select a new solution
new_solution = HEBBO(1000, 10).initialize_population()

# Evolve the population
population = HEBBO(1000, 10).initialize_population()
for _ in range(10):
    population = HEBBO(1000, 10).evolve_population(population)

# Print the final solution
print("Final Solution:", selected_solution.__call__(selected_solution.search_space))
print("New Solution:", new_solution.__call__(new_solution.search_space))