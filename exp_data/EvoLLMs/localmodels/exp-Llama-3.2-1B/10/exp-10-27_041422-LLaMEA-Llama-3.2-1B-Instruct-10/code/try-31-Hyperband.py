# Description: Hyperband Algorithm for Black Box Optimization
# Code: 
import numpy as np
import random

class Hyperband:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.best_func = None
        self.best_func_evals = 0
        self.sample_size = 1
        self.sample_dir = None
        self.sample_history = []

    def __call__(self, func):
        if self.budget <= 0:
            raise ValueError("Budget cannot be zero or negative")
        
        if self.best_func is not None:
            return self.best_func
        
        # Initialize the best function and its evaluation count
        self.best_func = func
        self.best_func_evals = 1
        
        # Set the sample size and directory
        self.sample_size = 10
        self.sample_dir = f"sample_{self.sample_size}"
        
        # Perform adaptive sampling
        for _ in range(self.budget):
            # Generate a random sample of size self.sample_size
            sample = np.random.uniform(-5.0, 5.0, size=self.dim)
            
            # Evaluate the function at the current sample
            func_eval = func(sample)
            
            # If this is the first evaluation, update the best function
            if self.best_func_evals == 1:
                self.best_func = func_eval
                self.best_func_evals = 1
            # Otherwise, update the best function if the current evaluation is better
            else:
                if func_eval > self.best_func:
                    self.best_func = func_eval
                    self.best_func_evals = 1
                else:
                    self.best_func_evals += 1
            
            # Save the current sample to the sample directory
            self.sample_history.append(sample)
        
        return self.best_func

class GeneticAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 100
        self.population_history = []
        self.best_individual = None
        self.best_fitness = float('inf')
        
    def __call__(self, func):
        if self.budget <= 0:
            raise ValueError("Budget cannot be zero or negative")
        
        if self.best_individual is not None:
            return self.best_individual
        
        # Initialize the population
        self.population = []
        for _ in range(self.population_size):
            individual = self.generate_individual(func, self.dim)
            self.population.append(individual)
        
        # Perform adaptive sampling
        for _ in range(self.budget):
            # Select the fittest individuals
            self.select_fittest(population)
            
            # Generate a new population
            new_population = []
            for _ in range(self.population_size):
                parent1, parent2 = random.sample(self.population, 2)
                child = self.crossover(parent1, parent2)
                new_population.append(child)
            
            # Replace the old population with the new one
            self.population = new_population
        
        # Evaluate the fitness of the new population
        fitness_values = [self.evaluate_fitness(individual, func) for individual, func in zip(self.population, func)]
        
        # Update the best individual and its fitness
        self.best_individual = self.population[np.argmax(fitness_values)]
        self.best_fitness = fitness_values[np.argmax(fitness_values)]
        
        return self.best_individual

    def generate_individual(self, func, dim):
        # Generate a random individual
        individual = np.random.uniform(-5.0, 5.0, size=dim)
        
        # Evaluate the function at the current individual
        fitness = func(individual)
        
        return individual, fitness

    def select_fittest(self, population):
        # Select the fittest individuals
        self.population_history.append(population)
        
        # Calculate the fitness of each individual
        fitness_values = [self.evaluate_fitness(individual, func) for individual, func in zip(population, func)]
        
        # Select the fittest individuals
        self.population = [individual for _, individual, fitness in zip(population, fitness_values, fitness_values) if fitness > 0.5]
        
    def crossover(self, parent1, parent2):
        # Perform crossover
        child = parent1[:len(parent1)//2] + parent2[len(parent1)//2:]
        
        return child

    def evaluate_fitness(self, individual, func):
        # Evaluate the fitness of the individual
        fitness = func(individual)
        
        return fitness

# Test the algorithm
func = lambda x: x**2
ga = GeneticAlgorithm(100, 2)
best_individual = ga(__call__(func))

print(f"Best individual: {best_individual}")
print(f"Best fitness: {ga.evaluate_fitness(best_individual, func)}")