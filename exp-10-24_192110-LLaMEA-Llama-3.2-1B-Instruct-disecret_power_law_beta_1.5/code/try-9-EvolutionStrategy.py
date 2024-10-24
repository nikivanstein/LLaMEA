import numpy as np
import random

class EvolutionStrategy:
    def __init__(self, budget, dim, mutation_rate=0.01):
        self.budget = budget
        self.dim = dim
        self.mutation_rate = mutation_rate
        self.population = self.initialize_population()

    def initialize_population(self):
        # Initialize the population with random solutions
        population = np.random.uniform(-5.0, 5.0, size=(self.budget, self.dim))
        return population

    def __call__(self, func):
        # Evaluate the function for each solution in the population
        scores = np.array([func(solution) for solution in self.population])

        # Select the top-scoring solutions
        top_indices = np.argsort(scores)[:self.budget]
        top_solutions = self.population[top_indices]

        # Create a new population by evolving the top solutions
        new_population = self.evolution(top_solutions, func, self.budget, self.mutation_rate)

        # Replace the old population with the new one
        self.population = new_population

        return scores

    def evolution(self, solutions, func, budget, mutation_rate):
        # Initialize the new population
        new_population = np.zeros((budget, self.dim))

        # Iterate over the solutions
        for i in range(budget):
            # Select two solutions to crossover
            parent1, parent2 = random.sample(solutions, 2)

            # Perform crossover
            child = np.concatenate((parent1[:self.dim//2], parent2[self.dim//2:]))

            # Mutate the child
            if random.random() < self.mutation_rate:
                child[random.randint(0, self.dim-1)] = random.uniform(-5.0, 5.0)

            # Add the child to the new population
            new_population[i] = child

        return new_population

# Example usage
def func(x):
    return x**2 + 2*x + 1

budget = 100
dim = 2
es = EvolutionStrategy(budget, dim)

# Initialize the population
es.population = es.initialize_population()

# Run the evolutionary algorithm
es.__call__(func)

# Print the final solution
print("Final solution:", np.argmax(es.population))
print("Final score:", np.max(np.array([func(solution) for solution in es.population])))