import numpy as np

class HEBBO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = np.linspace(-5.0, 5.0, dim)
        self.func_evaluations = 0
        self.population = []

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

    def mutate(self, individual):
        if np.random.rand() < 0.05:
            new_individual = individual + np.random.uniform(-0.1, 0.1, self.dim)
            return new_individual
        else:
            return individual

    def crossover(self, parent1, parent2):
        if np.random.rand() < 0.05:
            child = parent1[:np.random.randint(self.dim)]
            child.extend(parent2[np.random.randint(self.dim):])
            return child
        else:
            return parent1

    def evaluate_fitness(self, individual):
        func_value = self.__call__(individual)
        self.func_evaluations += 1
        return func_value

    def select_parents(self, num_parents):
        parents = np.random.choice(self.population, size=num_parents, replace=False)
        return parents

    def breed_parents(self, parents):
        offspring = []
        while len(offspring) < self.budget:
            parent1 = np.random.choice(parents)
            parent2 = np.random.choice(parents)
            child = self.crossover(parent1, parent2)
            offspring.append(child)
        return offspring

    def update_population(self, offspring):
        self.population = self.breed_parents(offspring)

# Example usage:
hebbbo = HEBBO(100, 5)
problem = lambda x: x**2  # noiseless function
best_individual = hebbbo.evaluate_fitness(problem)
print("Best individual:", best_individual)

# Refine the strategy
hebbbo.select_parents(20)
hebbbo.breed_parents(hebbbo.population)
hebbbo.update_population(hebbbo.population)

# Evaluate the new population
new_best_individual = hebbbo.evaluate_fitness(problem)
print("New best individual:", new_best_individual)