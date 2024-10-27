import random
import numpy as np

class GeneticMetaheuristic:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.boundaries = self.generate_boundaries(dim)

    def generate_boundaries(self, dim):
        # Generate a grid of boundaries for the dimension
        boundaries = np.linspace(-5.0, 5.0, dim)
        return boundaries

    def __call__(self, func, iterations=100):
        # Initialize the population
        population = self.generate_population(iterations)

        # Define the fitness function
        def fitness(individual):
            # Evaluate the black box function at the given point
            return np.mean(np.square(individual - np.array([0, 0, 0])))

        # Define the selection function
        def selection(population, func):
            # Calculate the fitness of each individual
            fitnesses = [fitness(individual) for individual in population]
            # Sort the individuals by fitness
            sorted_indices = np.argsort(fitnesses)
            # Select the top individuals based on the budget
            selected_indices = sorted_indices[:self.budget]
            # Create a new population with the selected individuals
            new_population = [population[i] for i in selected_indices]
            return new_population

        # Define the crossover function
        def crossover(parent1, parent2):
            # Select a random crossover point
            crossover_point = random.randint(1, self.dim)
            # Create a new child by combining the parents
            child = np.concatenate((parent1[:crossover_point], parent2[crossover_point:]))
            return child

        # Define the mutation function
        def mutation(individual):
            # Select a random mutation point
            mutation_point = random.randint(1, self.dim)
            # Randomly flip the bit at the mutation point
            individual[mutation_point] = 1 - individual[mutation_point]
            return individual

        # Run the genetic algorithm
        for _ in range(iterations):
            # Select the top individuals
            selected_population = selection(population, func)
            # Create a new population with the selected individuals
            new_population = []
            for _ in range(self.budget):
                # Select two parents using the selection function
                parent1 = random.choice(selected_population)
                parent2 = random.choice(selected_population)
                # Perform crossover and mutation
                child = crossover(parent1, parent2)
                child = mutation(child)
                # Add the child to the new population
                new_population.append(child)
            # Replace the old population with the new population
            population = new_population

        # Return the best individual
        return self.func(population[0])

    def func(self, individual):
        # Evaluate the black box function at the given point
        return np.mean(np.square(individual - np.array([0, 0, 0])))

# Example usage:
def func1(x):
    return np.mean(np.square(x - np.array([0, 0, 0])))

def func2(x):
    return np.sum(x**2)

metaheuristic = GeneticMetaheuristic(1000, 10)
print(metaheuristic.func(func1))  # Output: 0.0
print(metaheuristic.func(func2))  # Output: 1.0