import random
import numpy as np

class BlackBoxOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = np.linspace(-5.0, 5.0, dim)
        self.func_evaluations = 0

    def __call__(self, func, initial_individual, mutation_rate, num_generations):
        # Initialize the population with the initial individual
        population = [initial_individual]

        for _ in range(num_generations):
            # Select the next generation using the tournament selection method
            tournament_size = min(self.budget, len(population))
            winners = sorted([population[i] for i in random.sample(range(len(population)), tournament_size)], key=lambda x: x[func(x)], reverse=True)

            # Create the next generation by crossover and mutation
            new_individuals = []
            for winner in winners:
                parent1, parent2 = winner
                child = self.evaluate_fitness(parent1, func, parent2)
                new_individual = self.mutate(child, parent1, parent2, mutation_rate)
                new_individuals.append(new_individual)

            # Replace the old population with the new one
            population = new_individuals

        # Return the fittest individual in the new population
        return self.evaluate_fitness(population[0], func, population[0])

    def evaluate_fitness(self, individual, func, initial_individual):
        # Evaluate the function for the specified number of times
        num_evaluations = min(self.budget, self.func_evaluations + 1)
        func_evaluations = self.func_evaluations
        self.func_evaluations += num_evaluations

        # Generate a random point in the search space
        point = np.random.choice(self.search_space)

        # Evaluate the function at the point
        value = func(point)

        # Check if the function has been evaluated within the budget
        if value < 1e-10:  # arbitrary threshold
            # If not, return the current point as the optimal solution
            return point
        else:
            # If the function has been evaluated within the budget, return the point
            return point

    def mutate(self, individual, parent1, parent2, mutation_rate):
        # Generate a new individual by crossover and mutation
        if random.random() < mutation_rate:
            # Select a random point in the search space
            point = np.random.choice(self.search_space)

            # Evaluate the function at the point
            value = func(point)

            # Check if the function has been evaluated within the budget
            if value < 1e-10:  # arbitrary threshold
                # If not, return the current point as the optimal solution
                return point
            else:
                # If the function has been evaluated within the budget, return the point
                return point
        else:
            # If the individual has not been mutated, return it as is
            return individual

# One-line description: "Black Box Optimizer: A novel metaheuristic algorithm that efficiently solves black box optimization problems using a combination of random search and function evaluation"