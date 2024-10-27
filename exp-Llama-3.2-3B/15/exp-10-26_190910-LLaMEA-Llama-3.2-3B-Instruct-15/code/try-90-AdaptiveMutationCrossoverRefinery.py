import numpy as np
import random

class AdaptiveMutationCrossoverRefinery:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population = self.initialize_population()
        self.mutation_rate = 0.1
        self.crossover_rate = 0.7
        self.refine_probability = 0.15
        self.population_memory = []

    def initialize_population(self):
        population = np.random.uniform(-5.0, 5.0, (self.budget, self.dim))
        return population

    def __call__(self, func):
        for _ in range(self.budget):
            # Evaluate the population
            evaluations = func(self.population)

            # Store the best solution
            best_idx = np.argmin(evaluations)
            best_solution = self.population[best_idx]
            self.population_memory.append(best_solution)

            # Select parents
            parents = self.select_parents(evaluations)

            # Generate offspring
            offspring = self.generate_offspring(parents, evaluations)

            # Update the population
            self.population = np.vstack((self.population, offspring))

            # Apply mutation and crossover refinement
            self.population = self.apply_mutation_and_crossover_refinement(self.population, self.mutation_rate, self.crossover_rate, self.refine_probability)

        # Return the best solution found
        return self.population[np.argmin(evaluations)]

    def select_parents(self, evaluations):
        # Select parents based on the fitness
        parents = []
        for _ in range(self.dim):
            idx = np.random.choice(len(evaluations))
            parents.append(self.population[idx])
        return np.array(parents)

    def generate_offspring(self, parents, evaluations):
        # Generate offspring using crossover and mutation
        offspring = []
        for _ in range(self.dim):
            parent1, parent2 = random.sample(parents, 2)
            child = self.crossover(parent1, parent2)
            child = self.mutate(child)
            offspring.append(child)
        return np.array(offspring)

    def crossover(self, parent1, parent2):
        # Perform crossover using the crossover rate
        if random.random() < self.crossover_rate:
            child = (parent1 + parent2) / 2
            return child
        else:
            return parent1

    def mutate(self, solution):
        # Perform mutation using the mutation rate
        if random.random() < self.mutation_rate:
            idx = random.randint(0, self.dim - 1)
            solution[idx] += random.uniform(-1.0, 1.0)
            return solution
        else:
            return solution

    def apply_mutation_and_crossover_refinement(self, population, mutation_rate, crossover_rate, refine_probability):
        # Apply mutation and crossover refinement to the population
        new_population = population.copy()
        for i in range(len(population)):
            if random.random() < mutation_rate:
                new_population[i] = self.mutate(population[i])
            else:
                if random.random() < refine_probability:
                    new_population[i] = self.refine_individual(new_population[i])
                else:
                    new_population[i] = self.crossover(population[i], population[i])
        return new_population

    def refine_individual(self, individual):
        # Refine an individual by changing up to 15% of its components
        components = individual.tolist()
        for i in range(len(components)):
            if random.random() < 0.15:
                components[i] += random.uniform(-1.0, 1.0)
        return np.array(components)

# Example usage:
def bbb_function_1(x):
    return np.sum(x**2)

def bbb_function_2(x):
    return np.sum(x**3)

def bbb_function_3(x):
    return np.sum(np.sin(x))

def bbb_function_4(x):
    return np.sum(np.cos(x))

def bbb_function_5(x):
    return np.sum(x**4)

def bbb_function_6(x):
    return np.sum(np.exp(x))

def bbb_function_7(x):
    return np.sum(np.log(x))

def bbb_function_8(x):
    return np.sum(x**2 + x + 1)

def bbb_function_9(x):
    return np.sum(np.sin(x) + np.cos(x))

def bbb_function_10(x):
    return np.sum(x**2 + x + 1)

def bbb_function_11(x):
    return np.sum(np.exp(x) + np.log(x))

def bbb_function_12(x):
    return np.sum(x**2 + x + 1)

def bbb_function_13(x):
    return np.sum(np.sin(x) + np.cos(x))

def bbb_function_14(x):
    return np.sum(x**2 + x + 1)

def bbb_function_15(x):
    return np.sum(np.exp(x) + np.log(x))

def bbb_function_16(x):
    return np.sum(x**2 + x + 1)

def bbb_function_17(x):
    return np.sum(np.sin(x) + np.cos(x))

def bbb_function_18(x):
    return np.sum(x**2 + x + 1)

def bbb_function_19(x):
    return np.sum(np.exp(x) + np.log(x))

def bbb_function_20(x):
    return np.sum(x**2 + x + 1)

def bbb_function_21(x):
    return np.sum(np.sin(x) + np.cos(x))

def bbb_function_22(x):
    return np.sum(x**2 + x + 1)

def bbb_function_23(x):
    return np.sum(np.exp(x) + np.log(x))

def bbb_function_24(x):
    return np.sum(x**2 + x + 1)

# Initialize the algorithm
budget = 100
dim = 10
algorithm = AdaptiveMutationCrossoverRefinery(budget, dim)

# Evaluate the functions
functions = [bbb_function_1, bbb_function_2, bbb_function_3, bbb_function_4, bbb_function_5, 
             bbb_function_6, bbb_function_7, bbb_function_8, bbb_function_9, bbb_function_10,
             bbb_function_11, bbb_function_12, bbb_function_13, bbb_function_14, bbb_function_15,
             bbb_function_16, bbb_function_17, bbb_function_18, bbb_function_19, bbb_function_20,
             bbb_function_21, bbb_function_22, bbb_function_23, bbb_function_24]
for func in functions:
    algorithm(func)