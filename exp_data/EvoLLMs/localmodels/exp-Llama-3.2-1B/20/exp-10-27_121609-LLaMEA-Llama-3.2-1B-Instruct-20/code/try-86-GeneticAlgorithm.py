import numpy as np

class GeneticAlgorithm:
    def __init__(self, budget, dim, mutation_rate, population_size, bounds):
        self.budget = budget
        self.dim = dim
        self.mutation_rate = mutation_rate
        self.population_size = population_size
        self.bounds = bounds
        self.population = np.random.uniform(-5.0, 5.0, (self.population_size, self.dim))
        self.fitnesses = np.zeros((self.population_size, self.dim))

    def __call__(self, func):
        def objective(x):
            return func(x)

        def bounds(x):
            return (x.min() - 5.0, x.max() + 5.0)

        def mutate(individual):
            if np.random.rand() < self.mutation_rate:
                index1, index2 = np.random.randint(0, self.dim, 2)
                individual[index1], individual[index2] = individual[index2], individual[index1]

        def evaluate_fitness(individual):
            fitness = objective(individual)
            if fitness < self.fitnesses[individual, individual] + 1e-6:
                self.fitnesses[individual, individual] = fitness
                return individual
            else:
                return individual

        for _ in range(self.budget):
            for i in range(self.population_size):
                new_individual = evaluate_fitness(self.population[i])
                if new_individual is not None:
                    self.population[i] = new_individual

        return self.fitnesses

    def select_parents(self, fitnesses):
        return np.random.choice(self.population_size, size=len(fitnesses), replace=False, p=fitnesses / np.sum(fitnesses))

    def crossover(self, parent1, parent2):
        child = np.copy(parent1)
        for i in range(self.dim):
            if np.random.rand() < 0.5:
                child[i] = parent2[i]
        return child

    def mutate(self, individual):
        mutate(individual)

# BBOB test suite of 24 noiseless functions
# ```python
import numpy as np
from blackbox_optimization_bbb import BBOB

def objective(x):
    return np.sum(x**2)

def func1(x):
    return np.sin(x)

def func2(x):
    return np.cos(x)

def func3(x):
    return x**3

def func4(x):
    return x**4

def func5(x):
    return x**5

def func6(x):
    return np.sin(x) + np.cos(x)

def func7(x):
    return x**2 + np.sin(x)

def func8(x):
    return x**3 + np.cos(x)

def func9(x):
    return x**4 + np.sin(x)

def func10(x):
    return x**5 + np.cos(x)

def func11(x):
    return np.sin(x) + np.cos(x)

def func12(x):
    return x**2 + np.sin(x) + np.cos(x)

def func13(x):
    return x**3 + np.cos(x) + np.sin(x)

def func14(x):
    return x**4 + np.sin(x) + np.cos(x)

def func15(x):
    return x**5 + np.cos(x) + np.sin(x)

def func16(x):
    return np.sin(x) + np.cos(x) + x**2

def func17(x):
    return x**2 + np.sin(x) + np.cos(x) + x**3

def func18(x):
    return x**3 + np.cos(x) + np.sin(x) + x**4

def func19(x):
    return x**4 + np.sin(x) + np.cos(x) + x**5

def func20(x):
    return x**5 + np.cos(x) + np.sin(x) + x**6

def func21(x):
    return np.sin(x) + np.cos(x) + x**2 + np.sin(x) + np.cos(x)

def func22(x):
    return x**2 + np.sin(x) + np.cos(x) + x**3 + np.sin(x) + np.cos(x)

def func23(x):
    return x**3 + np.cos(x) + np.sin(x) + x**4 + np.sin(x) + np.cos(x)

def func24(x):
    return x**4 + np.sin(x) + np.cos(x) + x**5 + np.sin(x) + np.cos(x)

# Initialize the genetic algorithm
ga = GeneticAlgorithm(100, 5, 0.1, 100, [-5.0, 5.0])

# Run the genetic algorithm
fitnesses = ga(__call__, objective)

# Print the fitnesses
print("Fitnesses:", fitnesses)

# Select parents
parents = ga.select_parents(fitnesses)

# Crossover and mutate parents
for i in range(10):
    parents[i] = ga.crossover(parents[i], parents[(i+1)%10])

# Print the selected parents
print("Selected Parents:", parents)

# Evaluate fitnesses of the selected parents
fitnesses = ga(__call__, objective)
print("Fitnesses of Selected Parents:", fitnesses)

# Print the final solution
print("Final Solution:", parents[fitnesses.argmax()])