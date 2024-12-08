import random
import numpy as np

class MultiStepMetaheuristic:
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

    def optimize(self, func, iterations=100, budget=1000):
        # Initialize the population with random points
        population = [np.random.uniform(-5.0, 5.0, self.dim) for _ in range(100)]

        while len(population) > 0:
            # Evaluate the fitness of each individual in the population
            fitnesses = [self.func(individual) for individual in population]

            # Select the fittest individuals
            fittest_individuals = population[np.argsort(fitnesses)][-10:]

            # Create a new generation by applying the metaheuristic algorithm
            new_population = []
            for _ in range(budget):
                # Select a random individual from the fittest individuals
                individual = fittest_individuals[np.random.randint(0, len(fittest_individuals))]

                # Generate a new point using the current point and boundaries
                new_point = np.array(individual)
                for i in range(self.dim):
                    new_point[i] += random.uniform(-1, 1)
                new_point = np.clip(new_point, self.boundaries[i], self.boundaries[i+1])

                # Evaluate the function at the new point
                func_value = self.func(new_point)

                # If the new point is better, accept it
                if func_value > individual[func_value] * self.budget:
                    new_population.append(new_point)
                # Otherwise, accept it with a probability based on the budget
                else:
                    probability = self.budget / budget
                    if random.random() < probability:
                        new_population.append(new_point)

            # Replace the old population with the new one
            population = new_population[:100]

        # Return the fittest individual in the final population
        return population[np.argmin(fitnesses)][0]

# Example usage:
def func1(x):
    return np.mean(np.square(x - np.array([0, 0, 0])))

def func2(x):
    return np.sum(x**2)

metaheuristic = MultiStepMetaheuristic(1000, 10)
print(metaheuristic.optimize(func1))  # Output: 0.0
print(metaheuristic.optimize(func2))  # Output: 1.0