import random
import numpy as np
import matplotlib.pyplot as plt

class EvolutionaryBlackBoxOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 50
        self.population = self.init_population()
        self.fitness_scores = np.zeros((self.population_size, self.dim))
        self.search_spaces = [(-5.0, 5.0)] * self.dim

    def init_population(self):
        population = []
        for _ in range(self.population_size):
            individual = [random.uniform(self.search_spaces[i][0], self.search_spaces[i][1]) for i in range(self.dim)]
            population.append(individual)
        return population

    def __call__(self, func):
        def fitness(individual):
            return func(individual)

        def mutate(individual):
            if random.random() < 0.01:
                index = random.randint(0, self.dim - 1)
                self.search_spaces[index] = (self.search_spaces[index][0] + random.uniform(-1, 1), self.search_spaces[index][1] + random.uniform(-1, 1))
            return individual

        def evaluate(individual):
            return fitness(individual)

        for _ in range(self.budget):
            for i, individual in enumerate(self.population):
                fitness_scores[i] = fitness(individual)
                new_individual = evaluate(individual)
                if new_individual not in self.population[i]:
                    self.population[i].append(new_individual)
                    self.fitness_scores[i] = fitness(new_individual)

        return self.population

    def mutate_algorithm(self):
        for i in range(self.population_size):
            individual = self.population[i]
            if random.random() < 0.3:
                index = random.randint(0, self.dim - 1)
                self.search_spaces[index] = (self.search_spaces[index][0] + random.uniform(-1, 1), self.search_spaces[index][1] + random.uniform(-1, 1))
        return self.population

    def mutate_better_algorithm(self):
        for i in range(self.population_size):
            individual = self.population[i]
            if random.random() < 0.3:
                index = random.randint(0, self.dim - 1)
                self.search_spaces[index] = (self.search_spaces[index][0] + random.uniform(-1, 1), self.search_spaces[index][1] + random.uniform(-1, 1))
                if individual!= self.population[i]:
                    self.population[i].append(individual)
        return self.population

    def evaluate(self, func):
        return func(np.array(self.population))

# Test the algorithm
def test_black_box_optimization():
    # Define a test function
    def test_func(individual):
        return sum(individual)

    # Create an instance of the algorithm
    optimizer = EvolutionaryBlackBoxOptimizer(1000, 10)

    # Run the algorithm for 100 iterations
    for _ in range(100):
        optimizer.population = optimizer.mutate_algorithm()
        optimizer.population = optimizer.mutate_better_algorithm()

    # Evaluate the fitness of the best individual
    best_individual = optimizer.population[np.argmax([func(individual) for individual in optimizer.population])]
    best_fitness = test_func(best_individual)

    # Print the result
    print(f"Best individual: {best_individual}, Best fitness: {best_fitness}")

# Run the test
test_black_box_optimization()