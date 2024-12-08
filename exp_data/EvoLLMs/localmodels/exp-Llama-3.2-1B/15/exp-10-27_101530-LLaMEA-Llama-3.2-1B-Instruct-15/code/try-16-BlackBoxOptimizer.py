import random
import numpy as np

class BlackBoxOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = (-5.0, 5.0)
        self.func_evaluations = 0
        self.fitness_values = np.zeros((budget, dim))
        self.population = self.initialize_population()

    def initialize_population(self):
        # Initialize the population with random individuals
        population = []
        for _ in range(100):  # Initialize with 100 individuals
            individual = tuple(random.uniform(self.search_space[0], self.search_space[1]) for _ in range(self.dim))
            population.append(individual)
        return population

    def __call__(self, func):
        # Select the fittest individuals
        fittest_individuals = self.population[np.argsort(self.fitness_values, axis=0)]
        
        # Select individuals based on the probability of refinement
        probabilities = self.refine_probabilities(fittest_individuals)
        
        # Select the fittest individuals based on the probabilities
        selected_individuals = fittest_individuals[np.argsort(probabilities, axis=0)]
        
        # Evaluate the selected individuals
        fitness_values = np.array([func(individual) for individual in selected_individuals])
        
        # Update the population and fitness values
        self.population = selected_individuals
        self.fitness_values = fitness_values
        
        # Update the best individual
        best_individual = np.argmax(self.fitness_values)
        best_individual = self.population[best_individual]
        
        # Return the best individual
        return best_individual

    def refine_probabilities(self, individuals):
        # Define the mutation rate and the number of generations
        mutation_rate = 0.01
        num_generations = 100
        
        # Initialize the mutation rate and the population
        mutation_rate = np.clip(mutation_rate, 0, 0.5)
        population = individuals
        
        # Initialize the mutation rate and the fitness values
        fitness_values = np.zeros((num_generations, len(individuals)))
        
        # Run the selection process for the specified number of generations
        for _ in range(num_generations):
            # Select the fittest individuals
            fittest_individuals = population[np.argsort(fitness_values, axis=0)]
            
            # Select individuals based on the probability of mutation
            probabilities = self.mutation_probabilities(fittest_individuals)
            
            # Select the fittest individuals based on the probabilities
            selected_individuals = fittest_individuals[np.argsort(probabilities, axis=0)]
            
            # Evaluate the selected individuals
            fitness_values = np.array([func(individual) for individual in selected_individuals])
            
            # Update the population and fitness values
            population = selected_individuals
            fitness_values = fitness_values
        
        # Return the mutation probabilities
        return probabilities

    def mutation_probabilities(self, individuals):
        # Define the mutation rate and the number of generations
        mutation_rate = 0.01
        num_generations = 100
        
        # Initialize the mutation rate and the population
        mutation_rate = np.clip(mutation_rate, 0, 0.5)
        population = individuals
        
        # Initialize the mutation rate and the fitness values
        fitness_values = np.zeros((num_generations, len(individuals)))
        
        # Run the selection process for the specified number of generations
        for _ in range(num_generations):
            # Select the fittest individuals
            fittest_individuals = population[np.argsort(fitness_values, axis=0)]
            
            # Select individuals based on the probability of mutation
            probabilities = self.mutation_probabilities(fittest_individuals)
            
            # Select the fittest individuals based on the probabilities
            selected_individuals = fittest_individuals[np.argsort(probabilities, axis=0)]
            
            # Evaluate the selected individuals
            fitness_values = np.array([func(individual) for individual in selected_individuals])
            
            # Update the population and fitness values
            population = selected_individuals
            fitness_values = fitness_values
        
        # Return the mutation probabilities
        return probabilities

# Description: Novel Metaheuristic Algorithm for Black Box Optimization
# Code: 