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

            # Perform crossover and mutation
            mutated_population = []
            for individual in self.population:
                # Select two parents randomly
                parent1, parent2 = random.sample(self.population, 2)

                # Perform single-point crossover
                child = np.concatenate((parent1, parent2[1:]))

                # Perform Gaussian mutation with 50% probability
                if random.random() < 0.5:
                    child += np.random.normal(0, 1, child.shape)
                child = np.clip(child, -5.0, 5.0)

                mutated_population.append(child)
            self.population = np.array(mutated_population)

# Test the algorithm
def bbb_function(x):
    return x[0]**2 + x[1]**2 + x[2]**2

def bbb_function2(x):
    return x[0]**2 + x[1]**2 + x[2]**3

def bbb_function3(x):
    return x[0]**2 + x[1]**3 + x[2]**2

def bbb_function4(x):
    return x[0]**2 + x[1]**2 + x[2]**4

def bbb_function5(x):
    return x[0]**3 + x[1]**2 + x[2]**2

def bbb_function6(x):
    return x[0]**2 + x[1]**3 + x[2]**2

def bbb_function7(x):
    return x[0]**2 + x[1]**2 + x[2]**3

def bbb_function8(x):
    return x[0]**2 + x[1]**2 + x[2]**4

def bbb_function9(x):
    return x[0]**2 + x[1]**3 + x[2]**4

def bbb_function10(x):
    return x[0]**3 + x[1]**2 + x[2]**4

def bbb_function11(x):
    return x[0]**2 + x[1]**2 + x[2]**3

def bbb_function12(x):
    return x[0]**2 + x[1]**3 + x[2]**3

def bbb_function13(x):
    return x[0]**2 + x[1]**2 + x[2]**4

def bbb_function14(x):
    return x[0]**3 + x[1]**2 + x[2]**4

def bbb_function15(x):
    return x[0]**3 + x[1]**3 + x[2]**2

def bbb_function16(x):
    return x[0]**3 + x[1]**2 + x[2]**3

def bbb_function17(x):
    return x[0]**3 + x[1]**3 + x[2]**4

def bbb_function18(x):
    return x[0]**3 + x[1]**3 + x[2]**4

def bbb_function19(x):
    return x[0]**3 + x[1]**2 + x[2]**4

def bbb_function20(x):
    return x[0]**3 + x[1]**3 + x[2]**4

def bbb_function21(x):
    return x[0]**2 + x[1]**3 + x[2]**3

def bbb_function22(x):
    return x[0]**2 + x[1]**3 + x[2]**4

def bbb_function23(x):
    return x[0]**2 + x[1]**3 + x[2]**4

def bbb_function24(x):
    return x[0]**2 + x[1]**4 + x[2]**3

# Create an instance of the CDEA algorithm
cdea = CDEA(budget=100, dim=2)

# Define the function to optimize
def func(x):
    return bbb_function(x)

# Run the optimization
cdea(func)