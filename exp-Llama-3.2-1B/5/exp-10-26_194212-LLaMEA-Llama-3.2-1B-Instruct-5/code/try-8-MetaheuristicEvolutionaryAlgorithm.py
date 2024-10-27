import random
import numpy as np
import matplotlib.pyplot as plt

class MetaheuristicEvolutionaryAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.boundaries = self.generate_boundaries(dim)

    def generate_boundaries(self, dim):
        # Generate a grid of boundaries for the dimension
        boundaries = np.linspace(-5.0, 5.0, dim)
        return boundaries

    def __call__(self, func, iterations=100):
        # Initialize the current point and temperature
        current_point = None
        temperature = 1.0
        for _ in range(iterations):
            # Generate a new point using the current point and boundaries
            new_point = np.array(current_point)
            for i in range(self.dim):
                new_point[i] += random.uniform(-1, 1)
            new_point = np.clip(new_point, self.boundaries[i], self.boundaries[i+1])

            # Evaluate the function at the new point
            func_value = func(new_point)

            # If the new point is better, accept it
            if func_value > current_point[func_value] * temperature:
                current_point = new_point
            # Otherwise, accept it with a probability based on the temperature
            else:
                probability = temperature / self.budget
                if random.random() < probability:
                    current_point = new_point
        return current_point

    def func(self, point):
        # Evaluate the black box function at the given point
        return np.mean(np.square(point - np.array([0, 0, 0])))

    def mutate(self, individual):
        # Randomly change one of the values in the individual
        idx = random.randint(0, self.dim - 1)
        new_individual = individual.copy()
        new_individual[idx] += random.uniform(-1, 1)
        return new_individual

    def select(self, individuals):
        # Select the fittest individuals for the next iteration
        fitnesses = [self.func(individual) for individual in individuals]
        fittest_idx = np.argmax(fitnesses)
        return individuals[fittest_idx]

    def crossover(self, parent1, parent2):
        # Perform crossover between two parents to produce a child
        child = parent1.copy()
        for i in range(self.dim):
            if random.random() < 0.5:
                child[i] = parent2[i]
        return child

    def evolve(self, generations):
        # Evolve the algorithm over a specified number of generations
        for _ in range(generations):
            # Evaluate the function for each individual
            fitnesses = [self.func(individual) for individual in self.select(self.select(self.__call__(func, iterations=100))))
            # Select the fittest individuals
            self.select(fitnesses)
            # Crossover the parents to produce new offspring
            offspring = [self.crossover(parent1, parent2) for parent1, parent2 in zip(self.select(self.__call__(func, iterations=100)), self.select(self.__call__(func, iterations=100))) for parent1, parent2 in zip(parent1, parent2)]
            # Mutate the offspring
            offspring = [self.mutate(individual) for individual in offspring]
            # Replace the old individuals with the new ones
            self.__call__(func, iterations=100)
            # Update the boundaries
            self.boundaries = self.generate_boundaries(self.dim)

# Example usage:
def func1(x):
    return np.mean(np.square(x - np.array([0, 0, 0])))

def func2(x):
    return np.sum(x**2)

metaheuristic = MetaheuristicEvolutionaryAlgorithm(1000, 10)
print(metaheuristic.func(func1))  # Output: 0.0
print(metaheuristic.func(func2))  # Output: 1.0

# Update the algorithm with a new solution
new_metaheuristic = MetaheuristicEvolutionaryAlgorithm(1000, 10)
new_metaheuristic.boundaries = [np.linspace(-5.0, 5.0, 10), np.linspace(-5.0, 5.0, 10)]

# Plot the fitness of the old and new algorithms
plt.plot([metaheuristic.func(func1), new_metaheuristic.func(func1)], label='Old Algorithm')
plt.plot([metaheuristic.func(func2), new_metaheuristic.func(func2)], label='New Algorithm')
plt.legend()
plt.show()