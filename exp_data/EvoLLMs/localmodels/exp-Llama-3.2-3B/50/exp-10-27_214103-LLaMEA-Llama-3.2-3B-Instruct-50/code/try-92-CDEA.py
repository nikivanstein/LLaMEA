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
            mutated_population = self.mutate(self.population)
            crossovered_population = self.crossover(mutated_population)
            self.population = np.concatenate((crossovered_population, self.population[:self.population_size-self.population.shape[0]]))

    def crossover(self, population):
        # Perform single-point crossover with 50% rate
        offspring = []
        for _ in range(len(population)):
            if random.random() < 0.5:
                parent1, parent2 = random.sample(population, 2)
                child = np.concatenate((parent1, parent2[1:]))
            else:
                child = population[0]
            offspring.append(child)
        return np.array(offspring)

    def mutate(self, population):
        # Perform Gaussian mutation with 50% rate
        mutated_population = population + np.random.normal(0, 1, population.shape)
        return np.clip(mutated_population, -5.0, 5.0)

# Example usage:
if __name__ == "__main__":
    # Define the BBOB test suite functions
    functions = [
        lambda x: x[0]**2 + x[1]**2,
        lambda x: x[0]**3 - 3*x[0]**2*x[1] + x[1]**2,
        lambda x: x[0]**4 - 4*x[0]**3*x[1] + 3*x[0]**2*x[1]**2 + x[1]**4,
        lambda x: x[0]**5 - 5*x[0]**4*x[1] + 10*x[0]**3*x[1]**2 - 5*x[0]**2*x[1]**3 + x[1]**5,
        lambda x: x[0]**6 - 6*x[0]**5*x[1] + 15*x[0]**4*x[1]**2 - 10*x[0]**3*x[1]**3 + 5*x[0]**2*x[1]**4 + x[1]**6,
        lambda x: x[0]**7 - 7*x[0]**6*x[1] + 21*x[0]**5*x[1]**2 - 35*x[0]**4*x[1]**3 + 21*x[0]**3*x[1]**4 - 7*x[0]**2*x[1]**5 + x[1]**7,
        lambda x: x[0]**8 - 8*x[0]**7*x[1] + 56*x[0]**6*x[1]**2 - 112*x[0]**5*x[1]**3 + 56*x[0]**4*x[1]**4 - 28*x[0]**3*x[1]**5 + 8*x[0]**2*x[1]**6 + x[1]**8,
        lambda x: x[0]**9 - 9*x[0]**8*x[1] + 216*x[0]**7*x[1]**2 - 504*x[0]**6*x[1]**3 + 432*x[0]**5*x[1]**4 - 216*x[0]**4*x[1]**5 + 54*x[0]**3*x[1]**6 - 9*x[0]**2*x[1]**7 + x[1]**9,
        lambda x: x[0]**10 - 10*x[0]**9*x[1] + 720*x[0]**8*x[1]**2 - 1680*x[0]**7*x[1]**3 + 1680*x[0]**6*x[1]**4 - 720*x[0]**5*x[1]**5 + 240*x[0]**4*x[1]**6 - 40*x[0]**3*x[1]**7 + 10*x[0]**2*x[1]**8 + x[1]**10,
        lambda x: x[0]**11 - 11*x[0]**10*x[1] + 2310*x[0]**9*x[1]**2 - 4620*x[0]**8*x[1]**3 + 4620*x[0]**7*x[1]**4 - 2310*x[0]**6*x[1]**5 + 660*x[0]**5*x[1]**6 - 110*x[0]**4*x[1]**7 + 11*x[0]**3*x[1]**8 + x[1]**11,
        lambda x: x[0]**12 - 12*x[0]**11*x[1] + 39960*x[0]**10*x[1]**2 - 95040*x[0]**9*x[1]**3 + 95040*x[0]**8*x[1]**4 - 39960*x[0]**7*x[1]**5 + 11880*x[0]**6*x[1]**6 - 1764*x[0]**5*x[1]**7 + 220*x[0]**4*x[1]**8 + 12*x[0]**3*x[1]**9 + x[1]**12,
    ]

    # Initialize the CDEA algorithm
    cdea = CDEA(50, 10)

    # Evaluate the CDEA algorithm on the BBOB test suite
    for func in functions:
        cdea(func)
        print(f"Function: {func.__name__}, Best Value: {np.min(cdea.population, axis=0)}")