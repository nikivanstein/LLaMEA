import numpy as np
import random
import matplotlib.pyplot as plt

class BBOBMetaheuristic:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.func_evals = 0
        self.refining_strategies = {
            'random': self.refine_random,
            'gradient': self.refine_gradient,
            'bounded': self.refine_bounded
        }
        self.refining_strategy = self.refining_strategies['bounded']

    def __call__(self, func):
        # Check if the function can be evaluated within the budget
        if self.func_evals >= self.budget:
            raise ValueError("Not enough evaluations left to optimize the function")

        # Evaluate the function within the budget
        func_evals = self.func_evals
        self.func_evals += 1
        return func

    def search(self, func):
        # Define the search space
        bounds = np.linspace(-5.0, 5.0, self.dim, endpoint=False)
        
        # Initialize the solution
        sol = None
        
        # Try different initializations
        for _ in range(10):
            # Randomly initialize the solution
            sol = np.random.uniform(bounds, size=self.dim)
            
            # Evaluate the function at the solution
            func_sol = self.__call__(func, sol)
            
            # Check if the solution is better than the current best
            if func_sol < self.__call__(func, sol):
                # Update the solution
                sol = sol
        
        # Return the best solution found
        return sol

    def refine_random(self, sol):
        # Randomly refine the solution by changing one bit at a time
        for _ in range(self.dim):
            bit_index = random.randint(0, self.dim - 1)
            sol[bit_index] = 1 - sol[bit_index]
        
        # Evaluate the function at the refined solution
        func_sol = self.__call__(func, sol)
        
        # Check if the solution is better than the current best
        if func_sol < self.__call__(func, sol):
            # Update the solution
            sol = sol
    
    def refine_gradient(self, sol):
        # Refine the solution by adding a small gradient to the current solution
        gradient = np.random.uniform(-0.01, 0.01, self.dim)
        sol += gradient
        
        # Evaluate the function at the refined solution
        func_sol = self.__call__(func, sol)
        
        # Check if the solution is better than the current best
        if func_sol < self.__call__(func, sol):
            # Update the solution
            sol = sol
    
    def refine_bounded(self, sol):
        # Refine the solution by changing the bounds of the current solution
        lower_bound = -5.0
        upper_bound = 5.0
        
        # Refine the lower bound
        lower_bound = np.clip(sol, lower_bound, upper_bound)
        
        # Refine the upper bound
        upper_bound = np.clip(sol, lower_bound, upper_bound)
        
        # Evaluate the function at the refined solution
        func_sol = self.__call__(func, sol)
        
        # Check if the solution is better than the current best
        if func_sol < self.__call__(func, sol):
            # Update the solution
            sol = sol
    
    def evaluate_fitness(self, individual):
        # Evaluate the fitness of the individual
        func_sol = self.__call__(func, individual)
        return func_sol

class GeneticProgramming(BBOBMetaheuristic):
    def __init__(self, budget, dim):
        super().__init__(budget, dim)
        self.population_size = 100
        self.population = self.generate_initial_population()
    
    def generate_initial_population(self):
        # Generate a population of random individuals
        population = []
        for _ in range(self.population_size):
            individual = np.random.uniform(-5.0, 5.0, self.dim)
            population.append(individual)
        
        return population

    def evaluate_fitness(self, individual):
        # Evaluate the fitness of the individual
        func_sol = self.evaluate_fitness(individual)
        return func_sol

    def mutate(self, individual):
        # Mutate an individual by changing one bit at a time
        mutated_individual = individual.copy()
        for _ in range(self.dim):
            bit_index = random.randint(0, self.dim - 1)
            mutated_individual[bit_index] = 1 - mutated_individual[bit_index]
        
        return mutated_individual

    def crossover(self, parent1, parent2):
        # Perform crossover between two parents
        child = np.copy(parent1)
        for _ in range(self.dim):
            if random.random() < 0.5:
                child[_] = parent2[_]
        
        return child

    def selection(self, population):
        # Select the fittest individuals
        fitnesses = [self.evaluate_fitness(individual) for individual in population]
        indices = np.argsort(fitnesses)
        selected_indices = indices[:self.population_size // 2]
        
        # Select the fittest individuals
        selected_population = [population[i] for i in selected_indices]
        
        return selected_population

    def run(self):
        # Run the genetic programming algorithm
        population = self.generate_initial_population()
        for generation in range(100):
            # Evaluate the fitness of each individual
            fitnesses = [self.evaluate_fitness(individual) for individual in population]
            indices = np.argsort(fitnesses)
            selected_indices = indices[:self.population_size // 2]
            
            # Select the fittest individuals
            selected_population = [population[i] for i in selected_indices]
            
            # Mutate the selected individuals
            mutated_population = [self.mutate(individual) for individual in selected_population]
            
            # Perform crossover and selection
            new_population = self.selection(mutated_population)
            
            # Replace the old population with the new population
            population = new_population
    
    def plot_results(self):
        # Plot the results
        plt.plot([self.evaluate_fitness(individual) for individual in self.population])
        plt.xlabel('Generation')
        plt.ylabel('Fitness')
        plt.title('Genetic Programming Results')
        plt.show()

# Description: Evolutionary Algorithm for Black Box Optimization using Genetic Programming with Refining Strategies
# Code: 