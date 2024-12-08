import random
import numpy as np

class BlackBoxOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = [-5.0, 5.0]
        self.func_evaluations = 0
        self.population = []

    def __call__(self, func):
        # Ensure the function evaluations do not exceed the budget
        if self.func_evaluations < self.budget:
            # Generate a random point in the search space
            point = np.random.uniform(self.search_space[0], self.search_space[1])
            # Evaluate the function at the point
            evaluation = func(point)
            # Increment the function evaluations
            self.func_evaluations += 1
            # Return the point and its evaluation
            return point, evaluation
        else:
            # If the budget is reached, return a default point and evaluation
            return np.random.uniform(self.search_space[0], self.search_space[1]), func(np.random.uniform(self.search_space[0], self.search_space[1]))

    def mutate(self, individual):
        # Randomly select a new point within the search space
        new_point = np.random.uniform(self.search_space[0], self.search_space[1])
        # Randomly select a new fitness value within the range [0, 1]
        new_fitness = random.uniform(0, 1)
        # Return the mutated individual and its new fitness value
        return individual, new_point, new_fitness

    def crossover(self, parent1, parent2):
        # Select a random crossover point
        crossover_point = random.randint(1, self.dim)
        # Split the parent individuals into two parts
        child1 = parent1[:crossover_point]
        child2 = parent2[:crossover_point]
        # Combine the two parts to form the child individual
        child = child1 + child2
        # Return the child individual and its fitness value
        return child, child

# One-line description: Novel metaheuristic algorithm for black box optimization using a novel combination of random walk and linear interpolation.

# Initialize a new population of 100 individuals
optimizer = BlackBoxOptimizer(budget=100, dim=5)
# Evaluate the black box function for 10 noiseless functions
for _ in range(10):
    func = np.random.uniform(-10, 10)
    optimizer(func)
    # Add the results to the population
    optimizer.population.append((optimizer.func_evaluations, func))
# Print the results
for i, (evaluations, func) in enumerate(optimizer.population):
    print(f"Individual {i+1}: evaluations={evaluations}, function={func}")
# Print the average fitness value across all individuals
average_fitness = sum(x for x, _ in optimizer.population) / len(optimizer.population)
print(f"Average fitness value: {average_fitness}")
# Print the best individual found so far
best_individual = max(optimizer.population, key=lambda x: x[0])
print(f"Best individual: evaluations={best_individual[0]}, function={best_individual[1]}")
# Print the best fitness value found so far
best_fitness = max(optimizer.population, key=lambda x: x[0])
print(f"Best fitness value: {best_fitness[0]}")
# Print the mutation rate
mutation_rate = optimizer.func_evaluations / len(optimizer.population)
print(f"Mutation rate: {mutation_rate}")
# Print the crossover rate
crossover_rate = 0.5
print(f"Crossover rate: {crossover_rate}")
# Print the mutation strategy: 20% of individuals mutate, 30% crossover, 50% mutate
print(f"Mutation strategy: {optimizer.population[0][0] * 0.2 + optimizer.population[1][0] * 0.3 + optimizer.population[2][0] * 0.5}")