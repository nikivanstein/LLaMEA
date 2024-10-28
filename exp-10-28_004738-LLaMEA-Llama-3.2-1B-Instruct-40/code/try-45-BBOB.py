import random
import numpy as np

class BBOB:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.funcs = self.generate_functions()

    def generate_functions(self):
        functions = []
        for _ in range(24):
            func = lambda x: np.random.uniform(-5.0, 5.0)
            functions.append(func)
        return functions

    def __call__(self, func, initial_individual, bounds, budget):
        # Initialize the population with random individuals
        population = [initial_individual] * self.budget

        # Evolve the population for the specified number of generations
        for _ in range(self.budget):
            # Evaluate the fitness of each individual
            fitnesses = [func(individual, bounds) for individual, func in zip(population, self.funcs)]

            # Select the fittest individuals
            selected_indices = np.argsort(fitnesses)[-self.budget:]
            selected_individuals = [population[i] for i in selected_indices]

            # Create a new generation by mutating the selected individuals
            new_population = []
            for _ in range(self.budget):
                # Select two parents from the selected individuals
                parent1, parent2 = random.sample(selected_individuals, 2)

                # Create a new individual by combining the parents
                child = parent1[:len(parent1)//2] + parent2[len(parent1)//2:]

                # Mutate the child with a probability of 0.4
                if random.random() < 0.4:
                    # Generate a new child by adding a random noise to the child's genes
                    child = np.random.uniform(-5.0, 5.0, size=len(child)) + np.random.uniform(-0.1, 0.1, size=len(child))
                new_population.append(child)

            # Replace the old population with the new generation
            population = new_population

        # Return the fittest individual in the new population
        return max(population, key=fitnesses)[-1]

def f(x, bounds):
    return x**2 + 0.5*x + 0.1

def f_prime(x, bounds):
    return 2*x + 0.5

def f_double_prime(x, bounds):
    return 2

def f_double_prime_prime(x, bounds):
    return 4

def bbo_opt(func, initial_individual, bounds, budget):
    return BBOB(budget, len(initial_individual)).__call__(func, initial_individual, bounds, budget)