import random
import numpy as np

class BlackBoxOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = (-5.0, 5.0)
        self.population_size = 100
        self.crossover_rate = 0.5
        self.mutation_rate = 0.1

    def __call__(self, func):
        # Evaluate the function with the given budget
        func_evaluations = np.array([func(x) for x in np.random.uniform(self.search_space[0], self.search_space[1], self.population_size)])
        
        # Select the top-performing individuals
        top_individuals = np.argsort(func_evaluations)[-self.population_size:]
        
        # Create a new population by crossover and mutation
        new_population = []
        for _ in range(self.population_size):
            parent1, parent2 = random.sample(top_individuals, 2)
            child = (parent1 + parent2) / 2
            if random.random() < self.mutation_rate:
                child = random.uniform(self.search_space[0], self.search_space[1])
            new_population.append(child)
        
        # Replace the old population with the new one
        self.population = new_population
        
        # Evaluate the new population
        new_func_evaluations = np.array([func(x) for x in np.random.uniform(self.search_space[0], self.search_space[1], len(new_population))])
        
        # Return the best individual
        best_individual = np.argmax(new_func_evaluations)
        return new_population[best_individual]

    def fitness(self, individual):
        func_evaluations = np.array([self.func(x) for x in np.random.uniform(self.search_space[0], self.search_space[1], self.population_size)])
        return func_evaluations

# Black Box Optimizer function
def func(x):
    return np.sin(x)

# Genetic Algorithm
class GeneticAlgorithm:
    def __init__(self, population_size, crossover_rate, mutation_rate, budget):
        self.population_size = population_size
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.budget = budget
        self.population = self.initialize_population()
        self.fitnesses = self.initialize_fitnesses()

    def initialize_population(self):
        return [BlackBoxOptimizer(self.budget, dim) for dim in range(self.population_size)]

    def initialize_fitnesses(self):
        return [0.0 for _ in range(self.population_size)]

    def evaluate_fitness(self, individual):
        func_evaluations = self.fitness(individual)
        return func_evaluations

    def evaluate_population(self):
        new_fitnesses = []
        for _ in range(self.budget):
            new_population = []
            for i in range(self.population_size):
                parent1, parent2 = random.sample(self.population[i].population, 2)
                child = (parent1 + parent2) / 2
                if random.random() < self.mutation_rate:
                    child = random.uniform(self.search_space[0], self.search_space[1])
                new_population.append(child)
            new_fitnesses.extend(self.fitnesses(new_population))
            self.population = new_population
            self.fitnesses = new_fitnesses
        return new_fitnesses

    def select_top(self, top_fitnesses):
        return sorted(range(self.population_size), key=lambda i: top_fitnesses[i], reverse=True)[:self.population_size//2]

    def crossover(self, parent1, parent2):
        child = (parent1 + parent2) / 2
        if random.random() < self.crossover_rate:
            child = random.uniform(self.search_space[0], self.search_space[1])
        return child

    def mutate(self, individual):
        if random.random() < self.mutation_rate:
            return random.uniform(self.search_space[0], self.search_space[1])
        return individual

    def run(self):
        top_fitnesses = self.evaluate_population()
        best_individual = self.select_top(top_fitnesses)
        best_func = self.func(best_individual)
        return best_individual, best_func, top_fitnesses

# Run the genetic algorithm
ga = GeneticAlgorithm(population_size=100, crossover_rate=0.5, mutation_rate=0.1, budget=1000)
best_individual, best_func, top_fitnesses = ga.run()

# Print the results
print("Best Individual:", best_individual)
print("Best Function:", best_func)
print("Top Fitnesses:", top_fitnesses)
print("Best Fitness:", top_fitnesses[-1])
print("Best Individual Fitness:", best_fitnesses[best_individual])

# Update the Black Box Optimizer
ga.population[best_individual].fitness = top_fitnesses[best_individual]

# Print the updated Black Box Optimizer
print("Updated Black Box Optimizer:")
for i in range(len(ga.population)):
    print("Individual", i+1, ":", ga.population[i].fitness)