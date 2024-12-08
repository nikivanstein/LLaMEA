import random
import numpy as np

class BlackBoxOptimizer:
    """
    A metaheuristic algorithm to optimize black box functions.
    
    The algorithm uses a combination of genetic algorithm and simulated annealing to find the optimal solution.
    It handles a wide range of tasks and can be adapted to various optimization problems.
    """
    
    def __init__(self, budget, dim):
        """
        Initialize the optimizer with a budget and dimensionality.
        
        Args:
            budget (int): The maximum number of function evaluations allowed.
            dim (int): The dimensionality of the search space.
        """
        self.budget = budget
        self.dim = dim
        self.search_space = [(-5.0, 5.0)] * dim
        self.func_evaluations = 0
        self.population = None
        self.best_individual = None
        self.best_cost = float('inf')
        self.temperature = 1.0
    
    def __call__(self, func):
        """
        Optimize the black box function using the optimizer.
        
        Args:
            func (function): The black box function to optimize.
        
        Returns:
            tuple: The optimal solution and its cost.
        """
        
        # Initialize the solution and cost
        solution = None
        cost = float('inf')
        
        # Perform genetic algorithm
        for _ in range(self.budget):
            # Initialize the population
            self.population = self.initialize_population()
            
            # Run the genetic algorithm for the specified number of iterations
            for _ in range(self.budget):
                # Evaluate the fitness of each individual
                fitness = [self.evaluate_fitness(individual) for individual in self.population]
                
                # Select the fittest individuals
                self.population = self.select_fittest_individuals(fitness)
                
                # Perturb the fittest individuals
                self.population = self.perturb_individuals(self.population)
                
                # Evaluate the fitness of each individual again
                fitness = [self.evaluate_fitness(individual) for individual in self.population]
                
                # Calculate the new temperature
                self.temperature *= 0.9
                
                # If the new temperature is less than 0.3, accept the new solution
                if self.temperature < 0.3:
                    solution = self.population[fitness.index(min(fitness))]
                    cost = fitness[fitness.index(min(fitness))]
                    break
        
        # Run simulated annealing
        while self.temperature > 0.3:
            # Initialize the solution and cost
            solution = None
            cost = float('inf')
            
            # Run the simulated annealing loop
            for _ in range(self.budget):
                # Evaluate the fitness of each individual
                fitness = [self.evaluate_fitness(individual) for individual in self.population]
                
                # Select the fittest individuals
                self.population = self.select_fittest_individuals(fitness)
                
                # Perturb the fittest individuals
                self.population = self.perturb_individuals(self.population)
                
                # Evaluate the fitness of each individual again
                fitness = [self.evaluate_fitness(individual) for individual in self.population]
                
                # Calculate the new temperature
                self.temperature *= 0.9
                
                # If the new temperature is less than 0.3, accept the new solution
                if self.temperature < 0.3:
                    solution = self.population[fitness.index(min(fitness))]
                    cost = fitness[fitness.index(min(fitness))]
                    break
        
        # Return the optimal solution and its cost
        return solution, cost
    
    def initialize_population(self):
        """
        Initialize the population with random individuals.
        
        Returns:
            list: The initialized population.
        """
        
        # Initialize the population with random individuals
        population = [np.random.uniform(-5.0, 5.0, self.dim) for _ in range(100)]
        
        return population
    
    def select_fittest_individuals(self, fitness):
        """
        Select the fittest individuals based on their fitness.
        
        Args:
            fitness (list): The fitness of each individual.
        
        Returns:
            list: The selected fittest individuals.
        """
        
        # Select the fittest individuals based on their fitness
        selected_individuals = sorted(range(len(fitness)), key=lambda i: fitness[i], reverse=True)
        
        # Return the selected fittest individuals
        return selected_individuals
    
    def perturb_individuals(self, individuals):
        """
        Perturb the fittest individuals.
        
        Args:
            individuals (list): The fittest individuals.
        
        Returns:
            list: The perturbed individuals.
        """
        
        # Perturb the fittest individuals
        perturbed_individuals = [individual + (random.uniform(-1, 1), random.uniform(-1, 1)) for individual in individuals]
        
        return perturbed_individuals
    
    def evaluate_fitness(self, individual):
        """
        Evaluate the fitness of an individual.
        
        Args:
            individual (tuple): The individual.
        
        Returns:
            float: The fitness of the individual.
        """
        
        # Evaluate the fitness of the individual
        fitness = np.sum(np.abs(individual - self.search_space))
        
        return fitness
    
    def perturb_individual(self, individual):
        """
        Perturb the individual.
        
        Args:
            individual (tuple): The individual.
        
        Returns:
            tuple: The perturbed individual.
        """
        
        # Perturb the individual
        perturbed_individual = (individual[0] + random.uniform(-1, 1), individual[1] + random.uniform(-1, 1))
        
        return perturbed_individual
    
    def run(self, func, num_iterations):
        """
        Run the optimizer for a specified number of iterations.
        
        Args:
            func (function): The black box function to optimize.
            num_iterations (int): The number of iterations to run.
        
        Returns:
            tuple: The optimal solution and its cost.
        """
        
        # Run the optimizer for the specified number of iterations
        for _ in range(num_iterations):
            solution, cost = self(func)
            self.func_evaluations += 1
            
            # If the optimizer has reached the budget, break the loop
            if self.func_evaluations >= self.budget:
                break
        
        return solution, cost

# Description: A novel hybrid algorithm combining Genetic Algorithm and Simulated Annealing for efficient optimization of black box functions.
# Code: 