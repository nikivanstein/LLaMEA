import numpy as np
import os
from typing import Dict

class Hyperband:
    def __init__(self, budget: int, dim: int):
        """
        Initialize the Hyperband algorithm.

        Args:
        budget (int): The maximum number of function evaluations allowed.
        dim (int): The dimensionality of the search space.
        """
        self.budget = budget
        self.dim = dim
        self.best_func = None
        self.best_func_evals = 0
        self.sample_size = 1
        self.sample_dir = None

    def __call__(self, func: callable) -> callable:
        """
        Optimize the black box function using the Hyperband algorithm.

        Args:
        func (callable): The black box function to optimize.

        Returns:
        callable: The optimized function.
        """
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
            np.save(f"{self.sample_dir}_{self.sample_size}_{_}", sample)
        
        # Refine the strategy using evolutionary algorithms
        self.refine_strategy()
        
        return self.best_func

    def refine_strategy(self) -> None:
        """
        Refine the strategy using evolutionary algorithms.
        """
        # Define the population size, mutation rate, and number of generations
        population_size = 100
        mutation_rate = 0.01
        num_generations = 100
        
        # Initialize the population
        self.population = [self.evaluate_fitness(np.random.uniform(-5.0, 5.0, size=self.dim)) for _ in range(population_size)]
        
        # Evolve the population using genetic algorithms
        for _ in range(num_generations):
            # Select the fittest individuals
            fittest_individuals = self.population[np.argsort(self.population)]
            
            # Mutate the fittest individuals
            mutated_individuals = [self.evaluate_fitness(individual) for individual in fittest_individuals]
            for _ in range(len(fittest_individuals)):
                mutated_individuals[_] *= 2
            
            # Replace the fittest individuals with the mutated ones
            self.population = fittest_individuals[:population_size] + mutated_individuals[:population_size]
        
        # Update the best function
        self.best_func = self.population[np.argmax(self.population)][0]
        
        # Save the best function
        np.save(f"best_func_{self.sample_size}_{_}", self.best_func)