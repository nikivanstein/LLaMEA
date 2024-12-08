import numpy as np
import random

class HyperEvolutionOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 50
        self.mutate_prob = 0.1
        self.crossover_prob = 0.8
        self.fitness_func = None

    def __call__(self, func):
        # Initialize population
        population = np.random.uniform(-5.0, 5.0, (self.pop_size, self.dim))

        # Evaluate initial population
        fitness = np.array([func(x) for x in population])

        # Hyper-Evolutionary loop
        for i in range(self.budget):
            # Select top 20% of population with highest fitness
            sorted_indices = np.argsort(fitness)
            parents = population[sorted_indices[:int(0.2 * self.pop_size)]]

            # Perform Differential Evolution crossover
            offspring = np.zeros((self.pop_size, self.dim))
            for j in range(self.pop_size):
                parent1, parent2 = random.sample(parents, 2)
                mu = parent1 + (parent2 - parent1) * np.random.uniform(-1, 1)
                offspring[j] = mu

            # Perform Gaussian mutation
            mutated_offspring = offspring.copy()
            for j in range(self.pop_size):
                if random.random() < self.mutate_prob:
                    mutated_offspring[j] += np.random.normal(0, 1)
                    mutated_offspring[j] = np.clip(mutated_offspring[j], -5.0, 5.0)

            # Evaluate offspring
            new_fitness = np.array([func(x) for x in mutated_offspring])

            # Replace worst individuals
            population = np.concatenate((population[sorted_indices[int(0.2 * self.pop_size):]], mutated_offspring))

            # Update fitness
            fitness = np.concatenate((fitness, new_fitness))

        # Return best individual
        return population[np.argmin(fitness)]

# Example usage
def bbb_function1(x):
    return x[0]**2 + x[1]**2

def bbb_function2(x):
    return x[0]**2 + x[1]**2 + x[2]**2

def bbb_function3(x):
    return x[0]**2 + x[1]**2 + x[2]**2 + x[3]**2

def bbb_function4(x):
    return x[0]**2 + x[1]**2 + x[2]**2 + x[3]**2 + x[4]**2

def bbb_function5(x):
    return x[0]**2 + x[1]**2 + x[2]**2 + x[3]**2 + x[4]**2 + x[5]**2

def bbb_function6(x):
    return x[0]**2 + x[1]**2 + x[2]**2 + x[3]**2 + x[4]**2 + x[5]**2 + x[6]**2

def bbb_function7(x):
    return x[0]**2 + x[1]**2 + x[2]**2 + x[3]**2 + x[4]**2 + x[5]**2 + x[6]**2 + x[7]**2

def bbb_function8(x):
    return x[0]**2 + x[1]**2 + x[2]**2 + x[3]**2 + x[4]**2 + x[5]**2 + x[6]**2 + x[7]**2 + x[8]**2

def bbb_function9(x):
    return x[0]**2 + x[1]**2 + x[2]**2 + x[3]**2 + x[4]**2 + x[5]**2 + x[6]**2 + x[7]**2 + x[8]**2 + x[9]**2

def bbb_function10(x):
    return x[0]**2 + x[1]**2 + x[2]**2 + x[3]**2 + x[4]**2 + x[5]**2 + x[6]**2 + x[7]**2 + x[8]**2 + x[9]**2 + x[10]**2

def bbb_function11(x):
    return x[0]**2 + x[1]**2 + x[2]**2 + x[3]**2 + x[4]**2 + x[5]**2 + x[6]**2 + x[7]**2 + x[8]**2 + x[9]**2 + x[10]**2 + x[11]**2

def bbb_function12(x):
    return x[0]**2 + x[1]**2 + x[2]**2 + x[3]**2 + x[4]**2 + x[5]**2 + x[6]**2 + x[7]**2 + x[8]**2 + x[9]**2 + x[10]**2 + x[11]**2 + x[12]**2

def bbb_function13(x):
    return x[0]**2 + x[1]**2 + x[2]**2 + x[3]**2 + x[4]**2 + x[5]**2 + x[6]**2 + x[7]**2 + x[8]**2 + x[9]**2 + x[10]**2 + x[11]**2 + x[12]**2 + x[13]**2

def bbb_function14(x):
    return x[0]**2 + x[1]**2 + x[2]**2 + x[3]**2 + x[4]**2 + x[5]**2 + x[6]**2 + x[7]**2 + x[8]**2 + x[9]**2 + x[10]**2 + x[11]**2 + x[12]**2 + x[13]**2 + x[14]**2

def bbb_function15(x):
    return x[0]**2 + x[1]**2 + x[2]**2 + x[3]**2 + x[4]**2 + x[5]**2 + x[6]**2 + x[7]**2 + x[8]**2 + x[9]**2 + x[10]**2 + x[11]**2 + x[12]**2 + x[13]**2 + x[14]**2 + x[15]**2

def bbb_function16(x):
    return x[0]**2 + x[1]**2 + x[2]**2 + x[3]**2 + x[4]**2 + x[5]**2 + x[6]**2 + x[7]**2 + x[8]**2 + x[9]**2 + x[10]**2 + x[11]**2 + x[12]**2 + x[13]**2 + x[14]**2 + x[15]**2 + x[16]**2

def bbb_function17(x):
    return x[0]**2 + x[1]**2 + x[2]**2 + x[3]**2 + x[4]**2 + x[5]**2 + x[6]**2 + x[7]**2 + x[8]**2 + x[9]**2 + x[10]**2 + x[11]**2 + x[12]**2 + x[13]**2 + x[14]**2 + x[15]**2 + x[16]**2 + x[17]**2

def bbb_function18(x):
    return x[0]**2 + x[1]**2 + x[2]**2 + x[3]**2 + x[4]**2 + x[5]**2 + x[6]**2 + x[7]**2 + x[8]**2 + x[9]**2 + x[10]**2 + x[11]**2 + x[12]**2 + x[13]**2 + x[14]**2 + x[15]**2 + x[16]**2 + x[17]**2 + x[18]**2

def bbb_function19(x):
    return x[0]**2 + x[1]**2 + x[2]**2 + x[3]**2 + x[4]**2 + x[5]**2 + x[6]**2 + x[7]**2 + x[8]**2 + x[9]**2 + x[10]**2 + x[11]**2 + x[12]**2 + x[13]**2 + x[14]**2 + x[15]**2 + x[16]**2 + x[17]**2 + x[18]**2 + x[19]**2

def bbb_function20(x):
    return x[0]**2 + x[1]**2 + x[2]**2 + x[3]**2 + x[4]**2 + x[5]**2 + x[6]**2 + x[7]**2 + x[8]**2 + x[9]**2 + x[10]**2 + x[11]**2 + x[12]**2 + x[13]**2 + x[14]**2 + x[15]**2 + x[16]**2 + x[17]**2 + x[18]**2 + x[19]**2 + x[20]**2

def bbb_function21(x):
    return x[0]**2 + x[1]**2 + x[2]**2 + x[3]**2 + x[4]**2 + x[5]**2 + x[6]**2 + x[7]**2 + x[8]**2 + x[9]**2 + x[10]**2 + x[11]**2 + x[12]**2 + x[13]**2 + x[14]**2 + x[15]**2 + x[16]**2 + x[17]**2 + x[18]**2 + x[19]**2 + x[20]**2 + x[21]**2

def bbb_function22(x):
    return x[0]**2 + x[1]**2 + x[2]**2 + x[3]**2 + x[4]**2 + x[5]**2 + x[6]**2 + x[7]**2 + x[8]**2 + x[9]**2 + x[10]**2 + x[11]**2 + x[12]**2 + x[13]**2 + x[14]**2 + x[15]**2 + x[16]**2 + x[17]**2 + x[18]**2 + x[19]**2 + x[20]**2 + x[21]**2 + x[22]**2

def bbb_function23(x):
    return x[0]**2 + x[1]**2 + x[2]**2 + x[3]**2 + x[4]**2 + x[5]**2 + x[6]**2 + x[7]**2 + x[8]**2 + x[9]**2 + x[10]**2 + x[11]**2 + x[12]**2 + x[13]**2 + x[14]**2 + x[15]**2 + x[16]**2 + x[17]**2 + x[18]**2 + x[19]**2 + x[20]**2 + x[21]**2 + x[22]**2 + x[23]**2

def bbb_function24(x):
    return x[0]**2 + x[1]**2 + x[2]**2 + x[3]**2 + x[4]**2 + x[5]**2 + x[6]**2 + x[7]**2 + x[8]**2 + x[9]**2 + x[10]**2 + x[11]**2 + x[12]**2 + x[13]**2 + x[14]**2 + x[15]**2 + x[16]**2 + x[17]**2 + x[18]**2 + x[19]**2 + x[20]**2 + x[21]**2 + x[22]**2 + x[23]**2 + x[24]**2

# Example usage
if __name__ == "__main__":
    algorithm = HyperEvolutionOptimizer(100, 10)
    func = bbb_function1
    result = algorithm(func)
    print(result)