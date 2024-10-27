import random
import numpy as np

class GeneticBlackBoxOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = np.linspace(-5.0, 5.0, dim)
        self.func_evaluations = 0

    def __call__(self, func):
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

    def genetic_algorithm(self, population_size, mutation_rate, max_generations):
        # Initialize the population with random solutions
        population = self.evaluate_fitness(np.random.uniform(self.search_space, size=self.dim, size=population_size))

        for generation in range(max_generations):
            # Calculate the fitness of each individual
            fitnesses = self.evaluate_fitness(population)

            # Select the fittest individuals
            fittest_individuals = self.select_fittest(population, fitnesses)

            # Create a new population by crossover and mutation
            new_population = self.crossover(fittest_individuals, fitnesses)
            new_population = self.mutate(new_population, mutation_rate)

            # Replace the old population with the new one
            population = new_population

            # Print the fitness of each individual
            print(f"Generation {generation+1}, Fitnesses: {fitnesses}")

        # Return the fittest individual
        return self.evaluate_fitness(population)

    def crossover(self, parents, fitnesses):
        # Perform crossover between two parents
        offspring = []
        while len(offspring) < len(parents):
            parent1, parent2 = random.sample(parents, 2)
            child = (parent1 + parent2) / 2
            if random.random() < 0.5:
                child = parent2
            offspring.append(child)

        return offspring

    def mutate(self, population, mutation_rate):
        # Perform mutation on each individual
        mutated_population = []
        for individual in population:
            if random.random() < mutation_rate:
                mutated_individual = individual + random.uniform(-1, 1)
                mutated_individual = np.clip(mutated_individual, self.search_space[0], self.search_space[1])
                mutated_population.append(mutated_individual)
            else:
                mutated_population.append(individual)

        return mutated_population

    def evaluate_fitness(self, individual):
        # Evaluate the fitness of an individual
        func = lambda x: individual[0]*x[0] + individual[1]*x[1] + individual[2]*x[2]
        value = func(individual)
        return value

# One-line description: "Genetic Black Box Optimizer: A novel metaheuristic algorithm that efficiently solves black box optimization problems using a combination of genetic algorithm and function evaluation"
# Code: 