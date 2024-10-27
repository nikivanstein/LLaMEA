# Description: Novel Metaheuristic Algorithm for Black Box Optimization
# Code: 
# ```python
import numpy as np
import random

class BBOBBlackBoxOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = np.linspace(-5.0, 5.0, 100)
        self.func_evaluations = 0

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

    def select_strategy(self, individual, fitness):
        # Novel strategy: change individual lines of the selected solution to refine its strategy
        # 20% of the time, change the individual to a random line of the search space
        # 80% of the time, use the current individual
        if random.random() < 0.2:
            idx = random.randint(0, self.dim - 1)
            new_individual = individual[:idx] + [random.uniform(self.search_space[idx], self.search_space[idx + 1])] + individual[idx + 1:]
        else:
            new_individual = individual
        return new_individual, fitness

# Example usage:
optimizer = BBOBBlackBoxOptimizer(1000, 10)
func = lambda x: x**2
result = optimizer(func)
print(result)

# Initialize the population with the selected solution
population = [result]
for _ in range(100):
    new_individual, fitness = optimizer(func)
    population.append((new_individual, fitness))

# Print the population
print("Population:")
for individual, fitness in population:
    print(f"Individual: {individual}, Fitness: {fitness}")

# Evaluate the fitness of the best individual in the population
best_individual, best_fitness = max(population, key=lambda x: x[1])
print(f"Best Individual: {best_individual}, Best Fitness: {best_fitness}")

# Select a new individual based on the best individual's fitness
new_individual, new_fitness = optimizer.select_strategy(best_individual, best_fitness)
print(f"New Individual: {new_individual}, New Fitness: {new_fitness}")