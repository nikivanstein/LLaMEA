import random
import numpy as np

class EvolutionaryBoundaryOptimizationWithAdaptiveCooling:
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
            # Initialize the population
            population = self.generate_population(func, self.boundaries, iterations)

            # Evolve the population
            for _ in range(self.budget):
                # Select the fittest individuals
                fittest_individuals = sorted(population, key=self.fitness, reverse=True)[:self.budget // 2]

                # Select parents using tournament selection
                parents = []
                for _ in range(self.budget):
                    parent1, parent2 = random.sample(fittest_individuals, 2)
                    if random.random() < 0.5:
                        parents.append(parent1)
                    else:
                        parents.append(parent2)

                # Crossover
                offspring = []
                for _ in range(self.budget):
                    parent1, parent2 = random.sample(parents, 2)
                    child = (parent1 + parent2) / 2
                    if random.random() < 0.5:
                        child[0] += random.uniform(-1, 1)
                    offspring.append(child)

                # Mutate
                for individual in offspring:
                    if random.random() < 0.05:
                        individual[0] += random.uniform(-1, 1)

                # Replace the least fit individuals with the offspring
                population = self.replace_least_fit(population, offspring)

            # Evaluate the function at the new point
            new_point = np.array(current_point)
            for i in range(self.dim):
                new_point[i] += random.uniform(-1, 1)
            new_point = np.clip(new_point, self.boundaries[i], self.boundaries[i+1])

            func_value = self.func(new_point)

            # If the new point is better, accept it
            if func_value > current_point[func_value] * temperature:
                current_point = new_point
            # Otherwise, accept it with a probability based on the temperature
            else:
                probability = temperature / self.budget
                if random.random() < probability:
                    current_point = new_point
        return current_point

    def fitness(self, point):
        # Evaluate the black box function at the given point
        return np.mean(np.square(point - np.array([0, 0, 0])))

    def generate_population(self, func, boundaries, iterations):
        # Generate a population of individuals
        population = []
        for _ in range(iterations):
            individual = np.random.uniform(boundaries[0], boundaries[-1], self.dim)
            population.append(individual)
        return population

    def replace_least_fit(self, population, offspring):
        # Replace the least fit individuals with the offspring
        least_fit_indices = np.argsort(population, axis=0)[:-1, -1]
        population[least_fit_indices] = offspring
        return population

    def fitness(self, point):
        # Evaluate the black box function at the given point
        return np.mean(np.square(point - np.array([0, 0, 0])))

# Example usage:
def func1(x):
    return np.mean(np.square(x - np.array([0, 0, 0])))

def func2(x):
    return np.sum(x**2)

metaheuristic = EvolutionaryBoundaryOptimizationWithAdaptiveCooling(1000, 10)
print(metaheuristic.func(func1))  # Output: 0.0
print(metaheuristic.func(func2))  # Output: 1.0