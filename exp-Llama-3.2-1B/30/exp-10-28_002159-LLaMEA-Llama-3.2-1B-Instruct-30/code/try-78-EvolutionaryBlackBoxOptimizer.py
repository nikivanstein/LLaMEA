import random
import numpy as np

class EvolutionaryBlackBoxOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 50
        self.population = self.init_population()
        self.fitness_scores = np.zeros((self.population_size, self.dim))
        self.search_spaces = [(-5.0, 5.0)] * self.dim
        self.mutation_rate = 0.3

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

def genetic_algorithm(func, budget, dim):
    optimizer = EvolutionaryBlackBoxOptimizer(budget, dim)
    best_individual = optimizer(population)
    best_fitness = fitness(best_individual)
    print(f"Best Individual: {best_individual}, Best Fitness: {best_fitness}")
    return best_individual, best_fitness

def adaptive_mutation_rate(func, budget, dim):
    optimizer = EvolutionaryBlackBoxOptimizer(budget, dim)
    for _ in range(budget):
        for i, individual in enumerate(optimizer.population):
            fitness_scores = optimizer.fitness_scores
            best_individual = optimizer.population[np.argmax(fitness_scores)]
            new_individual = [random.uniform(self.search_spaces[j][0], self.search_spaces[j][1]) for j in range(self.dim)]
            new_individual = np.array(new_individual)
            if fitness(individual) > fitness(best_individual):
                optimizer.population[i] = new_individual
                optimizer.fitness_scores[i] = fitness(individual)
    return optimizer

# BBOB test suite functions
def test_func1(individual):
    return individual[0] + individual[1]

def test_func2(individual):
    return individual[0] * individual[1]

def test_func3(individual):
    return individual[0] - individual[1]

# Run genetic algorithm
best_individual, best_fitness = genetic_algorithm(test_func1, 100, 10)
print(f"Best Individual: {best_individual}, Best Fitness: {best_fitness}")

# Run adaptive mutation algorithm
optimizer = adaptive_mutation_rate(test_func1, 100, 10)
best_individual, best_fitness = optimizer(population)
print(f"Best Individual: {best_individual}, Best Fitness: {best_fitness}")