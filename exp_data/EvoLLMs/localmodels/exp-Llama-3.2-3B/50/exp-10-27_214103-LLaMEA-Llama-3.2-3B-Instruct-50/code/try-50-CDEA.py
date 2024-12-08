import numpy as np
import random

class CDEA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 50
        self.crowd_size = 10
        self.population = np.random.uniform(-5.0, 5.0, (self.population_size, self.dim))
        self.crowd = np.random.uniform(-5.0, 5.0, (self.crowd_size, self.dim))

    def __call__(self, func):
        for _ in range(self.budget):
            # Evaluate the population
            population_values = np.array([func(x) for x in self.population])

            # Evaluate the crowd
            crowd_values = np.array([func(x) for x in self.crowd])

            # Select the best individuals
            best_indices = np.argsort(population_values)[:, -self.crowd_size:]
            best_crowd_values = crowd_values[best_indices]

            # Select the worst individuals
            worst_indices = np.argsort(population_values)[:, :self.crowd_size]
            worst_population_values = population_values[worst_indices]

            # Update the population
            self.population = np.concatenate((best_crowd_values, worst_population_values))

            # Update the crowd
            self.crowd = self.population[:self.crowd_size]

            # Perform probabilistic crossover and mutation
            self.population = self.crossover(self.population)
            self.population = self.mutate(self.population)

    def crossover(self, population):
        # Perform single-point crossover with 50% probability
        offspring = []
        for _ in range(len(population)):
            if random.random() < 0.5:
                parent1, parent2 = random.sample(population, 2)
                child = np.concatenate((parent1, parent2[1:]))
                offspring.append(child)
            else:
                offspring.append(population[_])
        return np.array(offspring)

    def mutate(self, population):
        # Perform Gaussian mutation with 50% probability
        mutated_population = population + np.random.normal(0, 1, population.shape)
        return np.clip(mutated_population, -5.0, 5.0)

# Example usage
def bbb_function(x):
    return np.sum(x**2)

def evaluateBBOB(problem):
    cdea = CDEA(50, 10)
    cdea.func = bbb_function
    cdea()
    return cdea.population

# Run the BBOB test suite
import numpy as np
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Define the noiseless functions
def bbb_function(x):
    return np.sum(x**2)

def fun1(x):
    return np.sum(x**2)

def fun2(x):
    return np.sum(x**3)

def fun3(x):
    return np.sum(x**4)

def fun4(x):
    return np.sum(x**5)

def fun5(x):
    return np.sum(x**6)

def fun6(x):
    return np.sum(x**7)

def fun7(x):
    return np.sum(x**8)

def fun8(x):
    return np.sum(x**9)

def fun9(x):
    return np.sum(x**10)

def fun10(x):
    return np.sum(x**11)

def fun11(x):
    return np.sum(x**12)

def fun12(x):
    return np.sum(x**13)

def fun13(x):
    return np.sum(x**14)

def fun14(x):
    return np.sum(x**15)

def fun15(x):
    return np.sum(x**16)

def fun16(x):
    return np.sum(x**17)

def fun17(x):
    return np.sum(x**18)

def fun18(x):
    return np.sum(x**19)

def fun19(x):
    return np.sum(x**20)

def fun20(x):
    return np.sum(x**21)

def fun21(x):
    return np.sum(x**22)

def fun22(x):
    return np.sum(x**23)

def fun23(x):
    return np.sum(x**24)

def fun24(x):
    return np.sum(x**25)

# Run the BBOB test suite
budgets = [10, 20, 30, 40, 50]
dimensions = [10, 20, 30, 40, 50]

for budget in budgets:
    for dim in dimensions:
        cdea = CDEA(budget, dim)
        cdea.func = bbb_function
        cdea()
        print(f"Budget: {budget}, Dimension: {dim}, Population: {cdea.population}")
