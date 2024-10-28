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
            offspring = []
            for _ in range(len(self.population)):
                parent1, parent2 = random.sample(self.population, 2)
                if random.random() < 0.5:
                    child = np.concatenate((parent1, parent2[1:]))
                else:
                    child = np.concatenate((parent2, parent1[1:]))
                mutated_child = child + np.random.normal(0, 1, child.shape)
                offspring.append(np.clip(mutated_child, -5.0, 5.0))
            self.population = np.array(offspring)

# Test the algorithm
def bbb_function_1(x):
    return x[0]**2 + x[1]**2

def bbb_function_2(x):
    return x[0]**2 + x[1]**3

def bbb_function_3(x):
    return x[0]**3 + x[1]**2

def bbb_function_4(x):
    return x[0]**2 + x[1]**3 + x[2]**2

def bbb_function_5(x):
    return x[0]**2 + x[1]**2 + x[2]**3

def bbb_function_6(x):
    return x[0]**2 + x[1]**2 + x[2]**3 + x[3]**2

def bbb_function_7(x):
    return x[0]**2 + x[1]**3 + x[2]**2 + x[3]**3

def bbb_function_8(x):
    return x[0]**2 + x[1]**2 + x[2]**3 + x[3]**2 + x[4]**2

def bbb_function_9(x):
    return x[0]**2 + x[1]**3 + x[2]**2 + x[3]**3 + x[4]**2 + x[5]**2

def bbb_function_10(x):
    return x[0]**2 + x[1]**2 + x[2]**3 + x[3]**2 + x[4]**2 + x[5]**2 + x[6]**2

def bbb_function_11(x):
    return x[0]**2 + x[1]**3 + x[2]**2 + x[3]**3 + x[4]**2 + x[5]**2 + x[6]**2 + x[7]**2

def bbb_function_12(x):
    return x[0]**2 + x[1]**2 + x[2]**3 + x[3]**2 + x[4]**2 + x[5]**2 + x[6]**2 + x[7]**2 + x[8]**2

def bbb_function_13(x):
    return x[0]**2 + x[1]**3 + x[2]**2 + x[3]**3 + x[4]**2 + x[5]**2 + x[6]**2 + x[7]**2 + x[8]**2 + x[9]**2

def bbb_function_14(x):
    return x[0]**2 + x[1]**2 + x[2]**3 + x[3]**2 + x[4]**2 + x[5]**2 + x[6]**2 + x[7]**2 + x[8]**2 + x[9]**2 + x[10]**2

def bbb_function_15(x):
    return x[0]**2 + x[1]**3 + x[2]**2 + x[3]**3 + x[4]**2 + x[5]**2 + x[6]**2 + x[7]**2 + x[8]**2 + x[9]**2 + x[10]**2 + x[11]**2

def bbb_function_16(x):
    return x[0]**2 + x[1]**2 + x[2]**3 + x[3]**2 + x[4]**2 + x[5]**2 + x[6]**2 + x[7]**2 + x[8]**2 + x[9]**2 + x[10]**2 + x[11]**2 + x[12]**2

def bbb_function_17(x):
    return x[0]**2 + x[1]**3 + x[2]**2 + x[3]**3 + x[4]**2 + x[5]**2 + x[6]**2 + x[7]**2 + x[8]**2 + x[9]**2 + x[10]**2 + x[11]**2 + x[12]**2 + x[13]**2

def bbb_function_18(x):
    return x[0]**2 + x[1]**2 + x[2]**3 + x[3]**2 + x[4]**2 + x[5]**2 + x[6]**2 + x[7]**2 + x[8]**2 + x[9]**2 + x[10]**2 + x[11]**2 + x[12]**2 + x[13]**2 + x[14]**2

def bbb_function_19(x):
    return x[0]**2 + x[1]**3 + x[2]**2 + x[3]**3 + x[4]**2 + x[5]**2 + x[6]**2 + x[7]**2 + x[8]**2 + x[9]**2 + x[10]**2 + x[11]**2 + x[12]**2 + x[13]**2 + x[14]**2 + x[15]**2

def bbb_function_20(x):
    return x[0]**2 + x[1]**2 + x[2]**3 + x[3]**2 + x[4]**2 + x[5]**2 + x[6]**2 + x[7]**2 + x[8]**2 + x[9]**2 + x[10]**2 + x[11]**2 + x[12]**2 + x[13]**2 + x[14]**2 + x[15]**2 + x[16]**2

def bbb_function_21(x):
    return x[0]**2 + x[1]**3 + x[2]**2 + x[3]**3 + x[4]**2 + x[5]**2 + x[6]**2 + x[7]**2 + x[8]**2 + x[9]**2 + x[10]**2 + x[11]**2 + x[12]**2 + x[13]**2 + x[14]**2 + x[15]**2 + x[16]**2 + x[17]**2

def bbb_function_22(x):
    return x[0]**2 + x[1]**2 + x[2]**3 + x[3]**2 + x[4]**2 + x[5]**2 + x[6]**2 + x[7]**2 + x[8]**2 + x[9]**2 + x[10]**2 + x[11]**2 + x[12]**2 + x[13]**2 + x[14]**2 + x[15]**2 + x[16]**2 + x[17]**2 + x[18]**2

def bbb_function_23(x):
    return x[0]**2 + x[1]**3 + x[2]**2 + x[3]**3 + x[4]**2 + x[5]**2 + x[6]**2 + x[7]**2 + x[8]**2 + x[9]**2 + x[10]**2 + x[11]**2 + x[12]**2 + x[13]**2 + x[14]**2 + x[15]**2 + x[16]**2 + x[17]**2 + x[18]**2 + x[19]**2

def bbb_function_24(x):
    return x[0]**2 + x[1]**2 + x[2]**3 + x[3]**2 + x[4]**2 + x[5]**2 + x[6]**2 + x[7]**2 + x[8]**2 + x[9]**2 + x[10]**2 + x[11]**2 + x[12]**2 + x[13]**2 + x[14]**2 + x[15]**2 + x[16]**2 + x[17]**2 + x[18]**2 + x[19]**2 + x[20]**2

# Test the algorithm
algorithm = CDEA(100, 2)
algorithm(bbb_function_1)