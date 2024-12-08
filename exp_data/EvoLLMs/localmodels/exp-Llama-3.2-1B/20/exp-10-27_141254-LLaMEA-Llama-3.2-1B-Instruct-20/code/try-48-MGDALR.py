import numpy as np

class MGDALR:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.explore_rate = 0.1
        self.learning_rate = 0.01
        self.explore_count = 0
        self.max_explore_count = budget

    def __call__(self, func, logger):
        def inner(x):
            return func(x)
        
        # Initialize x to the lower bound
        x = np.array([-5.0] * self.dim)
        
        for _ in range(self.budget):
            # Evaluate the function at the current x
            y = inner(x)
            
            # If we've reached the maximum number of iterations, stop exploring
            if self.explore_count >= self.max_explore_count:
                break
            
            # If we've reached the upper bound, stop exploring
            if x[-1] >= 5.0:
                break
            
            # Learn a new direction using gradient descent
            learning_rate = self.learning_rate * (1 - self.explore_rate / self.max_explore_count)
            dx = -np.dot(x - inner(x), np.gradient(y))
            x += learning_rate * dx
            
            # Update the exploration count
            self.explore_count += 1
            
            # If we've reached the upper bound, stop exploring
            if x[-1] >= 5.0:
                break
        
        return x

class Individual:
    def __init__(self, dim):
        self.dim = dim
        self.fitness = 0

    def evaluate_fitness(self, func):
        self.fitness = func(self)

class Population:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.individuals = []

    def __call__(self, func, logger):
        for _ in range(self.budget):
            individual = Individual(self.dim)
            individual.fitness = func(individual)
            self.individuals.append(individual)

# Select a random individual from the population
population = Population(100, 10)
individual = population.individuals[np.random.randint(0, population.individuals.size)]

# Run the MGDALR algorithm
mgdalr = MGDALR(1000, 10)
solution = mgdalr(individual, population)

# Update the population with the selected solution
population.individuals = [individual for individual in population.individuals if individual.fitness > individual.fitness - 0.1]

# Print the updated population
print("Updated Population:")
for individual in population.individuals:
    print(individual.fitness)

# Print the updated individual
print(f"Updated Individual: x = {solution}")