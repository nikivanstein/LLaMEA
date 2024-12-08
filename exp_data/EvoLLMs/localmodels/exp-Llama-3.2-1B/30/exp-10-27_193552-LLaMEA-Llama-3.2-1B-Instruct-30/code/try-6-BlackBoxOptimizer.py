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
        
        # Run the optimizer for the specified number of iterations
        for _ in range(self.budget):
            # Evaluate the current solution
            solution = func(solution)
            
            # If the optimizer has reached the budget, break the loop
            if self.func_evaluations >= self.budget:
                break
            
            # Perturb the current solution
            self.perturb(solution)
        
        return solution, cost
    
    def perturb(self, solution):
        """
        Perturb the current solution.
        
        Args:
            solution (float): The current solution.
        
        Returns:
            float: The perturbed solution.
        """
        
        # Generate a random perturbation in the search space
        perturbation = (random.uniform(-1, 1), random.uniform(-1, 1))
        
        # Update the solution with the perturbation
        solution = (solution + perturbation[0], solution + perturbation[1])
        
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
        
        # Run the optimizer for the specified number of iterations
        for _ in range(num_iterations):
            solution, cost = self(func)
            self.func_evaluations += 1
            
            # If the optimizer has reached the budget, break the loop
            if self.func_evaluations >= self.budget:
                break
        
        return solution, cost

class TunedPerturbation(BlackBoxOptimizer):
    """
    A variant of the Randomized Black Box Optimization Algorithm with tuning of perturbation strategy.
    
    The perturbation strategy is tuned using a genetic algorithm to minimize the fitness of the individual lines of the solution.
    """
    
    def __init__(self, budget, dim):
        """
        Initialize the tuned perturbation optimizer with a budget and dimensionality.
        
        Args:
            budget (int): The maximum number of function evaluations allowed.
            dim (int): The dimensionality of the search space.
        """
        super().__init__(budget, dim)
        self.tuned_perturbation = True
        self.tuned_perturbation_strategy = None
        
    def perturb(self, solution):
        """
        Perturb the current solution using the tuned perturbation strategy.
        
        Args:
            solution (float): The current solution.
        
        Returns:
            float: The perturbed solution.
        """
        
        # Generate a random perturbation in the search space
        perturbation = (random.uniform(-1, 1), random.uniform(-1, 1))
        
        # Update the solution with the perturbation
        solution = (solution + perturbation[0], solution + perturbation[1])
        
        return solution

    def tune_perturbation(self, num_iterations):
        """
        Tune the perturbation strategy using a genetic algorithm.
        
        Args:
            num_iterations (int): The number of iterations to run.
        
        Returns:
            tuple: The optimal solution and its cost.
        """
        
        # Initialize the population of individuals
        population = [self.run(func, 100) for func in range(self.dim)]
        
        # Initialize the tournament selection
        tournament_size = 5
        tournament_selection = np.random.choice(population, size=tournament_size, replace=False)
        
        # Initialize the crossover and mutation operators
        crossover_operator = self.perturb
        mutation_operator = self.tuned_perturb
        
        # Run the genetic algorithm for the specified number of iterations
        for _ in range(num_iterations):
            # Initialize the new population
            new_population = []
            
            # Run the tournament selection
            for i in range(tournament_size):
                # Select an individual from the tournament
                individual = tournament_selection[i]
                
                # Perform crossover
                child = crossover_operator(individual)
                
                # Perform mutation
                child = mutation_operator(child)
                
                # Add the child to the new population
                new_population.append(child)
            
            # Replace the old population with the new population
            population = new_population
        
        # Return the optimal solution and its cost
        return self.run(func, 100)

# Description: Tuned Randomized Black Box Optimization Algorithm with Tuning of Perturbation Strategy
# Code: 