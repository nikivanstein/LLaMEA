import random
import numpy as np

class GeneticAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 50
        self.population = self.init_population()
        self.fitness_scores = np.zeros((self.population_size, self.dim))
        self.search_spaces = [(-5.0, 5.0)] * self.dim
        self.mutation_rate = 0.01

    def init_population(self):
        population = []
        for _ in range(self.population_size):
            individual = [random.uniform(self.search_spaces[i][0], self.search_spaces[i][1]) for i in range(self.dim)]
            population.append(individual)
        return population

    def __call__(self, func):
        def fitness(individual):
            return func(individual)

        for _ in range(self.budget):
            for i, individual in enumerate(self.population):
                fitness_scores[i] = fitness(individual)
            best_individual = self.population[np.argmax(fitness_scores)]
            new_individual = [random.uniform(self.search_spaces[j][0], self.search_spaces[j][1]) for j in range(self.dim)]
            new_individual = np.array(new_individual)
            if fitness(individual) > fitness(best_individual):
                self.population[i] = new_individual
                self.fitness_scores[i] = fitness(individual)

        return self.population

    def mutate(self, individual):
        if random.random() < self.mutation_rate:
            index = random.randint(0, self.dim - 1)
            self.search_spaces[index] = (self.search_spaces[index][0] + random.uniform(-1, 1), self.search_spaces[index][1] + random.uniform(-1, 1))
        return individual

    def evaluate(self, func):
        return func(np.array(self.population))

def fitnessBBOB(func, population):
    best_individual = population[np.argmax(func(population))]
    best_fitness = func(best_individual)
    return best_fitness

def mutate_bbb(func, individual, population):
    mutated_individual = individual.copy()
    mutated_individual[0] = random.uniform(func(0), func(1))
    mutated_individual[1] = random.uniform(func(0), func(1))
    return mutated_individual

# Initialize the Genetic Algorithm
ga = GeneticAlgorithm(100, 5)

# Define the Black Box function
def func(x):
    return x[0] + x[1] + x[2] + x[3] + x[4]

# Evaluate the Black Box function
fitness_bbb = fitnessBBOB(func, ga.population)
print(f"Black Box function: {fitness_bbb}")

# Run the Genetic Algorithm
ga.population = ga.init_population()
ga.evaluate(func)
best_individual = ga.population[np.argmax(ga.fitness_scores)]
best_fitness = fitnessBBOB(func, ga.population)

# Print the results
print(f"Best individual: {best_individual}")
print(f"Best fitness: {best_fitness}")

# Mutate the Genetic Algorithm
ga = GeneticAlgorithm(100, 5)
ga.population = ga.init_population()
ga.evaluate(func)
best_individual = ga.population[np.argmax(ga.fitness_scores)]
best_fitness = fitnessBBOB(func, ga.population)

# Print the results
print(f"Best individual after mutation: {best_individual}")
print(f"Best fitness after mutation: {best_fitness}")

# Run the Genetic Algorithm with mutation
ga.population = ga.init_population()
ga.evaluate(func)
best_individual = ga.population[np.argmax(ga.fitness_scores)]
best_fitness = fitnessBBOB(func, ga.population)

# Print the results
print(f"Best individual after mutation: {best_individual}")
print(f"Best fitness after mutation: {best_fitness}")

# Update the Genetic Algorithm
ga.population = ga.init_population()
ga.evaluate(func)
best_individual = ga.population[np.argmax(ga.fitness_scores)]
best_fitness = fitnessBBOB(func, ga.population)

# Print the results
print(f"Best individual: {best_individual}")
print(f"Best fitness: {best_fitness}")