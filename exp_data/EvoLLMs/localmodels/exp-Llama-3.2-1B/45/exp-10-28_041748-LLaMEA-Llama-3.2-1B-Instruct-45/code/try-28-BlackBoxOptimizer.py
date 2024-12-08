# Description: Evolutionary Algorithm for Black Box Optimization
# Code: 
# ```python
import random
import numpy as np

class BlackBoxOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = (-5.0, 5.0)
        self.population_size = 100
        self.crossover_rate = 0.5
        self.mutation_rate = 0.1
        self.population_history = []

    def __call__(self, func):
        # Evaluate the function with the given budget
        func_evaluations = np.array([func(x) for x in np.random.uniform(self.search_space[0], self.search_space[1], self.population_size)])
        
        # Select the top-performing individuals
        top_individuals = np.argsort(func_evaluations)[-self.population_size:]
        
        # Create a new population by crossover and mutation
        new_population = []
        for _ in range(self.population_size):
            parent1, parent2 = random.sample(top_individuals, 2)
            child = (parent1 + parent2) / 2
            if random.random() < self.mutation_rate:
                child = random.uniform(self.search_space[0], self.search_space[1])
            new_population.append(child)
        
        # Replace the old population with the new one
        self.population = new_population
        
        # Evaluate the new population
        new_func_evaluations = np.array([func(x) for x in np.random.uniform(self.search_space[0], self.search_space[1], len(new_population))])
        
        # Return the best individual
        best_individual = np.argmax(new_func_evaluations)
        return new_population[best_individual]

    def evolve(self):
        # Initialize a new population
        self.population = []
        
        # Create a new population by crossover and mutation
        for _ in range(self.population_size):
            parent1, parent2 = random.sample(self.population, 2)
            child = (parent1 + parent2) / 2
            if random.random() < self.mutation_rate:
                child = random.uniform(self.search_space[0], self.search_space[1])
            self.population.append(child)
        
        # Update the population history
        self.population_history.append((self.population, self.func_evaluations))

    def func_evaluations(self):
        return np.array([func(x) for x in np.random.uniform(self.search_space[0], self.search_space[1], len(self.population))])

    def select(self, top_n):
        # Select the top-performing individuals
        top_individuals = np.argsort(self.func_evaluations)[-top_n:]
        
        return top_individuals

    def mutate(self, individual):
        # Randomly change an individual
        if random.random() < self.mutation_rate:
            return random.uniform(self.search_space[0], self.search_space[1])
        return individual

    def crossover(self, parent1, parent2):
        # Perform crossover
        child = (parent1 + parent2) / 2
        return child

    def __str__(self):
        return f"BlackBoxOptimizer(budget={self.budget}, dim={self.dim})"


# Description: Evolutionary Algorithm for Black Box Optimization
# Code: 
# ```python
# BlackBoxOptimizer
# ```


# Initialize the optimizer
optimizer = BlackBoxOptimizer(1000, 10)

# Evolve the optimizer
optimizer.evolve()

# Print the final population and its fitness
best_individual = optimizer.population[0]
best_fitness = optimizer.func_evaluations()[0]

print(f"Best Individual: {best_individual}")
print(f"Best Fitness: {best_fitness}")

# Select the best individual
best_individual = optimizer.select(1)[0]

# Print the selected individual
print(f"Selected Individual: {best_individual}")

# Mutate the best individual
best_individual = optimizer.mutate(best_individual)

# Print the mutated individual
print(f"Mutated Individual: {best_individual}")

# Crossover the best individual with another individual
child = optimizer.crossover(best_individual, optimizer.population[1][0])

# Print the child individual
print(f"Child Individual: {child}")

# Print the final population
print(f"Final Population: {optimizer.population}")

# Print the final fitness
print(f"Final Fitness: {optimizer.func_evaluations()[0]}")