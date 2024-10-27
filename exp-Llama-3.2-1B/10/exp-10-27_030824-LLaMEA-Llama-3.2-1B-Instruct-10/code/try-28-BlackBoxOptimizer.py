import random
import numpy as np

class BlackBoxOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = [-5.0, 5.0]
        self.func_evaluations = 0
        self.best_individual = None
        self.best_fitness = float('-inf')
        self.population_size = 100

    def __call__(self, func, mutation_rate=0.01):
        # Ensure the function evaluations do not exceed the budget
        if self.func_evaluations < self.budget:
            # Generate a random point in the search space
            point = np.random.uniform(self.search_space[0], self.search_space[1])
            # Evaluate the function at the point
            evaluation = func(point)
            # Increment the function evaluations
            self.func_evaluations += 1
            # Return the point and its evaluation
            return point, evaluation
        else:
            # If the budget is reached, return a default point and evaluation
            return np.random.uniform(self.search_space[0], self.search_space[1]), func(np.random.uniform(self.search_space[0], self.search_space[1]))

    def mutate(self, individual):
        if random.random() < self.mutation_rate:
            # Randomly select two points in the search space
            idx1, idx2 = random.sample(range(self.dim), 2)
            # Swap the two points
            individual[idx1], individual[idx2] = individual[idx2], individual[idx1]
        return individual

    def crossover(self, parent1, parent2):
        # Select a random crossover point
        crossover_point = random.randint(1, self.dim - 1)
        # Create a new individual by combining the two parents
        child = [parent1[i] for i in range(crossover_point)] + [parent2[i] for i in range(crossover_point, self.dim)]
        return child

    def select_parents(self, num_parents):
        # Select random parents from the population
        parents = random.sample(self.population, num_parents)
        # Sort the parents by fitness
        parents.sort(key=self.fitness, reverse=True)
        return parents

    def fitness(self, individual):
        # Evaluate the function at the individual
        evaluation = self.func(individual)
        return evaluation

    def evaluate_fitness(self, individual, parents):
        # Combine the parents to form a new individual
        child = self.crossover(parents[0], parents[1])
        # Evaluate the function at the child
        evaluation = self.f(child)
        return evaluation

    def run(self, func, mutation_rate=0.01, num_parents=100, num_iterations=100):
        # Initialize the population
        population = [self.select_parents(num_parents) for _ in range(self.population_size)]
        # Run for the specified number of iterations
        for _ in range(num_iterations):
            # Evaluate the fitness of each individual
            fitnesses = [self.fitness(individual) for individual in population]
            # Select the fittest individuals
            parents = self.select_parents(len(population) // 2)
            # Select the remaining individuals to replace the fittest ones
            remaining = population[len(population) // 2:]
            # Replace the fittest individuals with the new parents
            population = [parents] + remaining
            # Mutate the population
            population = [self.mutate(individual) for individual in population]
            # Evaluate the fitness of each individual again
            fitnesses = [self.fitness(individual) for individual in population]
            # Update the best individual and its fitness
            self.best_individual = max(population, key=self.fitness)
            self.best_fitness = max(fitnesses, key=fitnesses)
            # Print the results
            print(f"Best Individual: {self.best_individual}, Best Fitness: {self.best_fitness}")
        return self.best_fitness

# One-line description: Novel metaheuristic algorithm for black box optimization using a novel combination of random walk and linear interpolation.

# Code: