import numpy as np

class HEBBO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = np.linspace(-5.0, 5.0, dim)
        self.func_evaluations = 0

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

class HBGX:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = np.linspace(-5.0, 5.0, dim)
        self.population_size = 50
        self.mutation_rate = 0.01
        self.population = np.random.uniform(-5.0, 5.0, (self.population_size, self.dim))

    def __call__(self, func):
        while self.func_evaluations < self.budget:
            self.population = np.random.uniform(self.search_space, size=(self.population_size, self.dim))
            fitness_values = self.evaluate_fitness(self.population, func)
            selected_individuals = self.select_top_individuals(fitness_values, self.population_size)
            new_individuals = self.mutate(selected_individuals, self.population_size)
            self.population = np.concatenate((self.population, new_individuals))
            self.func_evaluations += 1
            self.search_space = np.linspace(-5.0, 5.0, self.dim)
        return self.evaluate_fitness(self.population, func)

    def evaluate_fitness(self, population, func):
        fitness_values = np.array([func(individual) for individual in population])
        return fitness_values

    def select_top_individuals(self, fitness_values, population_size):
        top_individuals = np.argsort(fitness_values)[-population_size:]
        return top_individuals[:population_size]

    def mutate(self, individuals, population_size):
        mutated_individuals = np.random.choice(population_size, size=population_size, replace=True)
        return mutated_individuals

# One-line description with main idea
# Novel Metaheuristic Algorithm for Black Box Optimization
# This algorithm combines the strengths of HEBBO and HBGX, offering a more efficient and adaptive optimization strategy