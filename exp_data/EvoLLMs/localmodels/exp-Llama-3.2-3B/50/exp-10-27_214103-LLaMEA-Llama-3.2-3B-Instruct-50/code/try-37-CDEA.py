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

            # Perform crossover and mutation with probability 0.5
            mutated_population = self.population + np.random.normal(0, 1, self.population.shape)
            mutated_population = np.clip(mutated_population, -5.0, 5.0)
            crossover_mask = np.random.rand(self.population.shape[0], self.population.shape[1]) < 0.5
            mutated_population[crossover_mask] = self.population[~crossover_mask]
            self.population = mutated_population

# Test the algorithm
def bbb_1.1():
    return lambda x: x[0]**2 + 2*x[1]**2

def bbb_1.5():
    return lambda x: x[0]**2 + 3*x[1]**2

def bbb_2.1():
    return lambda x: 1 + 9*np.sin(x[0]) + 2*np.sin(x[1])

def bbb_2.3():
    return lambda x: 3 + 9*np.sin(x[0]) + 2*np.sin(x[1]) + np.sin(x[0])*np.sin(x[1])

def bbb_4():
    return lambda x: 2*x[0]**2 + 3*x[1]**2 + 0.1*np.sin(5*x[0]) + 0.1*np.sin(5*x[1])

def bbb_5():
    return lambda x: 1.5*x[0]**2 + 2.5*x[1]**2 + 0.5*np.sin(3*x[0]) + 0.5*np.sin(3*x[1])

def bbb_6():
    return lambda x: 2*x[0]**2 + 2*x[1]**2 + 0.5*np.sin(3*x[0]) + 0.5*np.sin(3*x[1])

def bbb_7():
    return lambda x: 2*x[0]**2 + 2*x[1]**2 + 0.5*np.sin(3*x[0]) + 0.5*np.sin(3*x[1]) + np.sin(x[0])*np.sin(x[1])

def bbb_8():
    return lambda x: 2*x[0]**2 + 2*x[1]**2 + 0.5*np.sin(3*x[0]) + 0.5*np.sin(3*x[1]) + np.sin(x[0])*np.sin(x[1]) + np.sin(2*x[0])*np.sin(2*x[1])

def bbb_9():
    return lambda x: 2*x[0]**2 + 2*x[1]**2 + 0.5*np.sin(3*x[0]) + 0.5*np.sin(3*x[1]) + np.sin(x[0])*np.sin(x[1]) + np.sin(2*x[0])*np.sin(2*x[1]) + np.sin(x[0])*np.sin(x[1])

def bbb_10():
    return lambda x: 2*x[0]**2 + 2*x[1]**2 + 0.5*np.sin(3*x[0]) + 0.5*np.sin(3*x[1]) + np.sin(x[0])*np.sin(x[1]) + np.sin(2*x[0])*np.sin(2*x[1]) + np.sin(x[0])*np.sin(x[1]) + np.sin(3*x[0])*np.sin(3*x[1])

def bbb_11():
    return lambda x: 2*x[0]**2 + 2*x[1]**2 + 0.5*np.sin(3*x[0]) + 0.5*np.sin(3*x[1]) + np.sin(x[0])*np.sin(x[1]) + np.sin(2*x[0])*np.sin(2*x[1]) + np.sin(x[0])*np.sin(x[1]) + np.sin(3*x[0])*np.sin(3*x[1]) + np.sin(2*x[0])*np.sin(2*x[1])

def bbb_12():
    return lambda x: 2*x[0]**2 + 2*x[1]**2 + 0.5*np.sin(3*x[0]) + 0.5*np.sin(3*x[1]) + np.sin(x[0])*np.sin(x[1]) + np.sin(2*x[0])*np.sin(2*x[1]) + np.sin(x[0])*np.sin(x[1]) + np.sin(3*x[0])*np.sin(3*x[1]) + np.sin(2*x[0])*np.sin(2*x[1]) + np.sin(x[0])*np.sin(x[1])

def bbb_13():
    return lambda x: 2*x[0]**2 + 2*x[1]**2 + 0.5*np.sin(3*x[0]) + 0.5*np.sin(3*x[1]) + np.sin(x[0])*np.sin(x[1]) + np.sin(2*x[0])*np.sin(2*x[1]) + np.sin(x[0])*np.sin(x[1]) + np.sin(3*x[0])*np.sin(3*x[1]) + np.sin(2*x[0])*np.sin(2*x[1]) + np.sin(x[0])*np.sin(x[1]) + np.sin(2*x[0])*np.sin(2*x[1])

def bbb_14():
    return lambda x: 2*x[0]**2 + 2*x[1]**2 + 0.5*np.sin(3*x[0]) + 0.5*np.sin(3*x[1]) + np.sin(x[0])*np.sin(x[1]) + np.sin(2*x[0])*np.sin(2*x[1]) + np.sin(x[0])*np.sin(x[1]) + np.sin(3*x[0])*np.sin(3*x[1]) + np.sin(2*x[0])*np.sin(2*x[1]) + np.sin(x[0])*np.sin(x[1]) + np.sin(2*x[0])*np.sin(2*x[1]) + np.sin(3*x[0])*np.sin(3*x[1])

def bbb_15():
    return lambda x: 2*x[0]**2 + 2*x[1]**2 + 0.5*np.sin(3*x[0]) + 0.5*np.sin(3*x[1]) + np.sin(x[0])*np.sin(x[1]) + np.sin(2*x[0])*np.sin(2*x[1]) + np.sin(x[0])*np.sin(x[1]) + np.sin(3*x[0])*np.sin(3*x[1]) + np.sin(2*x[0])*np.sin(2*x[1]) + np.sin(x[0])*np.sin(x[1]) + np.sin(2*x[0])*np.sin(2*x[1]) + np.sin(3*x[0])*np.sin(3*x[1]) + np.sin(2*x[0])*np.sin(2*x[1])

def bbb_16():
    return lambda x: 2*x[0]**2 + 2*x[1]**2 + 0.5*np.sin(3*x[0]) + 0.5*np.sin(3*x[1]) + np.sin(x[0])*np.sin(x[1]) + np.sin(2*x[0])*np.sin(2*x[1]) + np.sin(x[0])*np.sin(x[1]) + np.sin(3*x[0])*np.sin(3*x[1]) + np.sin(2*x[0])*np.sin(2*x[1]) + np.sin(x[0])*np.sin(x[1]) + np.sin(2*x[0])*np.sin(2*x[1]) + np.sin(3*x[0])*np.sin(3*x[1]) + np.sin(2*x[0])*np.sin(2*x[1]) + np.sin(x[0])*np.sin(x[1])

def bbb_17():
    return lambda x: 2*x[0]**2 + 2*x[1]**2 + 0.5*np.sin(3*x[0]) + 0.5*np.sin(3*x[1]) + np.sin(x[0])*np.sin(x[1]) + np.sin(2*x[0])*np.sin(2*x[1]) + np.sin(x[0])*np.sin(x[1]) + np.sin(3*x[0])*np.sin(3*x[1]) + np.sin(2*x[0])*np.sin(2*x[1]) + np.sin(x[0])*np.sin(x[1]) + np.sin(2*x[0])*np.sin(2*x[1]) + np.sin(3*x[0])*np.sin(3*x[1]) + np.sin(2*x[0])*np.sin(2*x[1]) + np.sin(x[0])*np.sin(x[1])

def bbb_18():
    return lambda x: 2*x[0]**2 + 2*x[1]**2 + 0.5*np.sin(3*x[0]) + 0.5*np.sin(3*x[1]) + np.sin(x[0])*np.sin(x[1]) + np.sin(2*x[0])*np.sin(2*x[1]) + np.sin(x[0])*np.sin(x[1]) + np.sin(3*x[0])*np.sin(3*x[1]) + np.sin(2*x[0])*np.sin(2*x[1]) + np.sin(x[0])*np.sin(x[1]) + np.sin(2*x[0])*np.sin(2*x[1]) + np.sin(3*x[0])*np.sin(3*x[1]) + np.sin(2*x[0])*np.sin(2*x[1]) + np.sin(x[0])*np.sin(x[1])

def bbb_19():
    return lambda x: 2*x[0]**2 + 2*x[1]**2 + 0.5*np.sin(3*x[0]) + 0.5*np.sin(3*x[1]) + np.sin(x[0])*np.sin(x[1]) + np.sin(2*x[0])*np.sin(2*x[1]) + np.sin(x[0])*np.sin(x[1]) + np.sin(3*x[0])*np.sin(3*x[1]) + np.sin(2*x[0])*np.sin(2*x[1]) + np.sin(x[0])*np.sin(x[1]) + np.sin(2*x[0])*np.sin(2*x[1]) + np.sin(3*x[0])*np.sin(3*x[1]) + np.sin(2*x[0])*np.sin(2*x[1]) + np.sin(x[0])*np.sin(x[1])

def bbb_20():
    return lambda x: 2*x[0]**2 + 2*x[1]**2 + 0.5*np.sin(3*x[0]) + 0.5*np.sin(3*x[1]) + np.sin(x[0])*np.sin(x[1]) + np.sin(2*x[0])*np.sin(2*x[1]) + np.sin(x[0])*np.sin(x[1]) + np.sin(3*x[0])*np.sin(3*x[1]) + np.sin(2*x[0])*np.sin(2*x[1]) + np.sin(x[0])*np.sin(x[1]) + np.sin(2*x[0])*np.sin(2*x[1]) + np.sin(3*x[0])*np.sin(3*x[1]) + np.sin(2*x[0])*np.sin(2*x[1]) + np.sin(x[0])*np.sin(x[1])

def bbb_21():
    return lambda x: 2*x[0]**2 + 2*x[1]**2 + 0.5*np.sin(3*x[0]) + 0.5*np.sin(3*x[1]) + np.sin(x[0])*np.sin(x[1]) + np.sin(2*x[0])*np.sin(2*x[1]) + np.sin(x[0])*np.sin(x[1]) + np.sin(3*x[0])*np.sin(3*x[1]) + np.sin(2*x[0])*np.sin(2*x[1]) + np.sin(x[0])*np.sin(x[1]) + np.sin(2*x[0])*np.sin(2*x[1]) + np.sin(3*x[0])*np.sin(3*x[1]) + np.sin(2*x[0])*np.sin(2*x[1]) + np.sin(x[0])*np.sin(x[1])

def bbb_22():
    return lambda x: 2*x[0]**2 + 2*x[1]**2 + 0.5*np.sin(3*x[0]) + 0.5*np.sin(3*x[1]) + np.sin(x[0])*np.sin(x[1]) + np.sin(2*x[0])*np.sin(2*x[1]) + np.sin(x[0])*np.sin(x[1]) + np.sin(3*x[0])*np.sin(3*x[1]) + np.sin(2*x[0])*np.sin(2*x[1]) + np.sin(x[0])*np.sin(x[1]) + np.sin(2*x[0])*np.sin(2*x[1]) + np.sin(3*x[0])*np.sin(3*x[1]) + np.sin(2*x[0])*np.sin(2*x[1]) + np.sin(x[0])*np.sin(x[1])

def bbb_23():
    return lambda x: 2*x[0]**2 + 2*x[1]**2 + 0.5*np.sin(3*x[0]) + 0.5*np.sin(3*x[1]) + np.sin(x[0])*np.sin(x[1]) + np.sin(2*x[0])*np.sin(2*x[1]) + np.sin(x[0])*np.sin(x[1]) + np.sin(3*x[0])*np.sin(3*x[1]) + np.sin(2*x[0])*np.sin(2*x[1]) + np.sin(x[0])*np.sin(x[1]) + np.sin(2*x[0])*np.sin(2*x[1]) + np.sin(3*x[0])*np.sin(3*x[1]) + np.sin(2*x[0])*np.sin(2*x[1]) + np.sin(x[0])*np.sin(x[1])

def bbb_24():
    return lambda x: 2*x[0]**2 + 2*x[1]**2 + 0.5*np.sin(3*x[0]) + 0.5*np.sin(3*x[1]) + np.sin(x[0])*np.sin(x[1]) + np.sin(2*x[0])*np.sin(2*x[1]) + np.sin(x[0])*np.sin(x[1]) + np.sin(3*x[0])*np.sin(3*x[1]) + np.sin(2*x[0])*np.sin(2*x[1]) + np.sin(x[0])*np.sin(x[1]) + np.sin(2*x[0])*np.sin(2*x[1]) + np.sin(3*x[0])*np.sin(3*x[1]) + np.sin(2*x[0])*np.sin(2*x[1]) + np.sin(x[0])*np.sin(x[1])

# Initialize the CDEA algorithm
cdea = CDEA(budget=100, dim=2)

# Evaluate the functions using the CDEA algorithm
def evaluate_bbb_bboB(bbb_func, cdea):
    values = []
    for i in range(24):
        x = np.random.uniform(-5.0, 5.0, (1, cdea.dim))
        y = cdea(x, bbb_func)
        values.append(y)
    return values

# Test the BBOB test suite
def bbb_1():
    return bbb_1.1

def bbb_2():
    return bbb_2.1

def bbb_3():
    return bbb_3.1

def bbb_4():
    return bbb_4()

def bbb_5():
    return bbb_5()

def bbb_6():
    return bbb_6()

def bbb_7():
    return bbb_7()

def bbb_8():
    return bbb_8()

def bbb_9():
    return bbb_9()

def bbb_10():
    return bbb_10()

def bbb_11():
    return bbb_11()

def bbb_12():
    return bbb_12()

def bbb_13():
    return bbb_13()

def bbb_14():
    return bbb_14()

def bbb_15():
    return bbb_15()

def bbb_16():
    return bbb_16()

def bbb_17():
    return bbb_17()

def bbb_18():
    return bbb_18()

def bbb_19():
    return bbb_19()

def bbb_20():
    return bbb_20()

def bbb_21():
    return bbb_21()

def bbb_22():
    return bbb_22()

def bbb_23():
    return bbb_23()

def bbb_24():
    return bbb_24()

# Evaluate the BBOB test suite using the CDEA algorithm
values = evaluate_bbb_bboB(bbb_1, cdea)
values.extend(evaluate_bbb_bboB(bbb_2, cdea))
values.extend(evaluate_bbb_bboB(bbb_3, cdea))
values.extend(evaluate_bbb_bboB(bbb_4, cdea))
values.extend(evaluate_bbb_bboB(bbb_5, cdea))
values.extend(evaluate_bbb_bboB(bbb_6, cdea))
values.extend(evaluate_bbb_bboB(bbb_7, cdea))
values.extend(evaluate_bbb_bboB(bbb_8, cdea))
values.extend(evaluate_bbb_bboB(bbb_9, cdea))
values.extend(evaluate_bbb_bboB(bbb_10, cdea))
values.extend(evaluate_bbb_bboB(bbb_11, cdea))
values.extend(evaluate_bbb_bboB(bbb_12, cdea))
values.extend(evaluate_bbb_bboB(bbb_13, cdea))
values.extend(evaluate_bbb_bboB(bbb_14, cdea))
values.extend(evaluate_bbb_bboB(bbb_15, cdea))
values.extend(evaluate_bbb_bboB(bbb_16, cdea))
values.extend(evaluate_bbb_bboB(bbb_17, cdea))
values.extend(evaluate_bbb_bboB(bbb_18, cdea))
values.extend(evaluate_bbb_bboB(bbb_19, cdea))
values.extend(evaluate_bbb_bboB(bbb_20, cdea))
values.extend(evaluate_bbb_bboB(bbb_21, cdea))
values.extend(evaluate_bbb_bboB(bbb_22, cdea))
values.extend(evaluate_bbb_bboB(bbb_23, cdea))
values.extend(evaluate_bbb_bboB(bbb_24, cdea))

# Print the results
for i, value in enumerate(values):
    print(f"Function {i+1}: {value[0]}")