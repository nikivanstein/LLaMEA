import numpy as np
import random
from scipy.optimize import minimize

class BBOBBlackBoxOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = np.linspace(-5.0, 5.0, 100)
        self.func_evaluations = 0
        self.population = self.generate_population()

    def generate_population(self):
        population = []
        for _ in range(100):
            individual = [random.uniform(self.search_space[i], self.search_space[i+1]) for i in range(self.dim)]
            population.append(individual)
        return population

    def __call__(self, func):
        def wrapper(x):
            if self.func_evaluations >= self.budget:
                raise Exception("Optimization limit reached")
            return func(x)

        self.func_evaluations += 1
        try:
            result = minimize(wrapper, self.search_space[0], method="SLSQP", bounds=[(x, x) for x in self.search_space])
            return result.x
        except Exception as e:
            print(f"Error: {e}")
            return None

    def select_next_generation(self, population):
        next_generation = []
        for _ in range(self.budget):
            fitness_values = [self.evaluate_fitness(individual) for individual in population]
            selected_indices = np.argsort(fitness_values)[-self.budget:]
            selected_population = [population[i] for i in selected_indices]
            next_generation.append(self.select_strategies(selected_population))
        return next_generation

    def select_strategies(self, population):
        # Probability of changing individual lines
        prob_change = 0.2
        # Change individual lines based on probability
        for individual in population:
            if random.random() < prob_change:
                individual[0], individual[1] = random.uniform(-5.0, 5.0), random.uniform(-5.0, 5.0)
        return population

    def evaluate_fitness(self, individual):
        func_value = individual[0]**2
        return func_value

# Example usage:
optimizer = BBOBBlackBoxOptimizer(1000, 10)
func = lambda x: x**2
result = optimizer(func)
print(result)