import random
import numpy as np

class BlackBoxOptimizer:
    """
    A metaheuristic algorithm to optimize black box functions.
    
    The algorithm uses a combination of random search and perturbation to find the optimal solution.
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
        
        # Perform random search
        for _ in range(self.budget):
            # Perturb the current solution
            perturbed_solution = self.perturb(solution)
            
            # Evaluate the new solution
            new_cost = func(perturbed_solution)
            
            # Update the solution and cost if the new solution is better
            if new_cost < cost:
                solution = perturbed_solution
                cost = new_cost
        
        return solution, cost
    
    def perturb(self, solution):
        """
        Perturb the current solution.
        
        Args:
            solution (tuple): The current solution.
        
        Returns:
            tuple: The perturbed solution.
        """
        
        # Generate a random perturbation in the search space
        perturbation = (random.uniform(-1, 1), random.uniform(-1, 1))
        
        # Update the solution with the perturbation
        solution = (solution[0] + perturbation[0], solution[1] + perturbation[1])
        
        return solution
    
    def run(self, func, num_iterations):
        """
        Run the optimizer for a specified number of iterations.
        
        Args:
            func (function): The black box function to optimize.
            num_iterations (int): The number of iterations to run.
        
        Returns:
            tuple: The optimal solution and its cost.
        """
        
        # Initialize the population of solutions
        population = [self.evaluate_fitness(func) for _ in range(100)]
        
        # Run the optimizer for the specified number of iterations
        for _ in range(num_iterations):
            # Select the fittest solution
            fittest_solution = population.index(max(population))
            
            # Create a new solution by refining the fittest solution
            new_solution = self.refine_solution(fittest_solution, population, func)
            
            # Add the new solution to the population
            population.append(new_solution)
        
        # Return the optimal solution and its cost
        return population[0], max(population)
    
    def refine_solution(self, solution, population, func):
        """
        Refine the solution using a genetic algorithm.
        
        Args:
            solution (tuple): The current solution.
            population (list): The population of solutions.
            func (function): The black box function to optimize.
        
        Returns:
            tuple: The refined solution.
        """
        
        # Initialize the population of offspring
        offspring = []
        
        # Generate offspring using crossover and mutation
        for _ in range(len(population)):
            parent1, parent2 = random.sample(population, 2)
            child = self.crossover(parent1, parent2)
            child = self.mutate(child)
            offspring.append(child)
        
        # Select the fittest offspring
        fittest_offspring = offspring.index(max(offspring))
        
        # Return the refined solution
        return offspring[fittest_offspring]
    
    def crossover(self, parent1, parent2):
        """
        Perform crossover between two parents.
        
        Args:
            parent1 (tuple): The first parent.
            parent2 (tuple): The second parent.
        
        Returns:
            tuple: The offspring.
        """
        
        # Generate two offspring
        child1 = (parent1[0] + parent2[0]) / 2, (parent1[1] + parent2[1]) / 2
        child2 = (parent1[0] - parent2[0]) / 2, (parent1[1] - parent2[1]) / 2
        
        # Return the offspring
        return child1, child2
    
    def mutate(self, solution):
        """
        Mutate a solution.
        
        Args:
            solution (tuple): The solution to mutate.
        
        Returns:
            tuple: The mutated solution.
        """
        
        # Generate a random mutation
        mutation = (random.uniform(-1, 1), random.uniform(-1, 1))
        
        # Return the mutated solution
        return (solution[0] + mutation[0], solution[1] + mutation[1])