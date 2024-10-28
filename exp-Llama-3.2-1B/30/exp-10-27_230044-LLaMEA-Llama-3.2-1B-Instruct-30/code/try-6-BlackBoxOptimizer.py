import numpy as np
import random
import math

class BlackBoxOptimizer:
    def __init__(self, budget, dim, learning_rate=0.01, mutation_rate=0.1, population_size=100, n_generations=100):
        self.budget = budget
        self.dim = dim
        self.learning_rate = learning_rate
        self.mutation_rate = mutation_rate
        self.population_size = population_size
        self.n_generations = n_generations
        self.population = self.initialize_population()

    def initialize_population(self):
        return [np.random.uniform(-5.0, 5.0, self.dim) for _ in range(self.population_size)]

    def __call__(self, func):
        """
        Optimize the black box function using Black Box Optimizer.

        Args:
            func (function): The black box function to optimize.

        Returns:
            float: The optimized value of the function.
        """
        # Initialize weights and bias using a neural network
        self.weights = np.random.rand(self.dim)
        self.bias = np.random.rand(1)
        self.weights = np.vstack((self.weights, [0]))
        self.bias = np.append(self.bias, 0)

        # Define the neural network architecture
        self.nn = {
            'input': self.dim,
            'hidden': self.dim,
            'output': 1
        }

        # Define the optimization function
        def optimize(x):
            # Forward pass
            y = np.dot(x, self.weights) + self.bias
            # Backward pass
            dy = np.dot(self.nn['output'].reshape(-1, 1), (y - func(x)))
            # Update weights and bias
            self.weights -= self.learning_rate * dy * x
            self.bias -= self.learning_rate * dy
            return y

        # Run the optimization algorithm
        for _ in range(self.n_generations):
            # Evaluate fitness of each individual
            fitnesses = [optimize(individual) for individual in self.population]

            # Select parents using tournament selection
            parents = self.select_parents(fitnesses, self.population_size // 2)

            # Create offspring using crossover
            offspring = self.crossover(parents)

            # Mutate offspring
            mutated_offspring = self.mutate(offspring, self.mutation_rate)

            # Replace parents with offspring
            self.population = mutated_offspring

        # Return the best individual
        return self.population[fitnesses.index(max(fitnesses))]

    def select_parents(self, fitnesses, num_parents):
        # Select parents using tournament selection
        parents = []
        for _ in range(num_parents):
            individual = random.choice(fitnesses)
            fitness = individual
            while True:
                parent = random.choice(fitnesses)
                if fitness < parent:
                    individual = parent
                    fitness = parent
                    break
            parents.append(individual)
        return parents

    def crossover(self, parents):
        # Create offspring using crossover
        offspring = []
        for _ in range(len(parents) // 2):
            parent1 = random.choice(parents)
            parent2 = random.choice(parents)
            offspring.append(self.evaluateBBOB(parent1, func))
            offspring.append(self.evaluateBBOB(parent2, func))
        return offspring

    def mutate(self, offspring, mutation_rate):
        # Mutate offspring
        mutated_offspring = []
        for individual in offspring:
            if random.random() < mutation_rate:
                mutated_individual = individual + np.random.uniform(-5.0, 5.0)
                mutated_individual = np.clip(mutated_individual, -5.0, 5.0)
                mutated_offspring.append(mutated_individual)
            else:
                mutated_offspring.append(individual)
        return mutated_offspring

# Define the function to be optimized
def func(x):
    return x**2 + 0.5 * x**3

# Create an instance of the Black Box Optimizer
optimizer = BlackBoxOptimizer(budget=1000, dim=2)

# Optimize the function
best_individual = optimizer(func)

# Print the result
print(f"Best individual: {best_individual}")
print(f"Best fitness: {best_individual}")