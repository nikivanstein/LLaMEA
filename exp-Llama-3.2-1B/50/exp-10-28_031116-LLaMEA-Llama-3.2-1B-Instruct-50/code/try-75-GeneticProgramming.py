import random
import numpy as np

class GeneticProgramming:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 100
        self.population = self.initialize_population()
        self.fitness_scores = np.zeros((self.population_size, self.dim))
        self.evolutionary_algorithm()

    def initialize_population(self):
        # Initialize the population with random solutions
        solutions = []
        for _ in range(self.population_size):
            solution = np.random.uniform(-5.0, 5.0, self.dim)
            solutions.append(solution)
        return np.array(solutions)

    def fitness_function(self, func, solution):
        # Evaluate the function at the solution
        return func(solution)

    def __call__(self, func):
        # Optimize the function using the evolutionary algorithm
        while self.fitness_scores.shape[0] < self.budget:
            # Select parents using tournament selection
            parents = self.select_parents()
            # Crossover to combine parents
            offspring = self.crossover(parents)
            # Mutate offspring
            offspring = self.mutate(offspring)
            # Evaluate fitness of offspring
            fitness_scores = self.fitness_function(func, offspring)
            # Update population with offspring
            self.population = self.update_population(parents, offspring, fitness_scores)
        # Return the best solution found
        return self.population[np.argmax(self.fitness_scores)]

    def select_parents(self):
        # Select parents using tournament selection
        parents = []
        for _ in range(self.population_size // 2):
            parent1 = np.random.uniform(-5.0, 5.0, self.dim)
            parent2 = np.random.uniform(-5.0, 5.0, self.dim)
            if np.random.rand() < 0.5:
                parent1 = parent2
            parents.append(parent1)
        return parents

    def crossover(self, parents):
        # Crossover to combine parents
        offspring = parents.copy()
        for _ in range(self.population_size // 2):
            index1 = random.randint(0, self.population_size - 1)
            index2 = random.randint(0, self.population_size - 1)
            if index1 < index2:
                offspring[index1], offspring[index2] = offspring[index2], offspring[index1]
        return offspring

    def mutate(self, offspring):
        # Mutate offspring
        for i in range(self.dim):
            if random.random() < 0.1:
                offspring[i] += random.uniform(-0.1, 0.1)
        return offspring

    def update_population(self, parents, offspring, fitness_scores):
        # Update population with offspring
        new_population = np.concatenate((parents, offspring))
        return new_population

# Example usage
def black_box_function(x):
    return x**2 + 2*x + 1

genetic_programming = GeneticProgramming(100, 10)
best_solution = genetic_programming(__call__(black_box_function))
print("Best solution:", best_solution)
print("Fitness score:", genetic_programming.fitness_function(black_box_function, best_solution))