import random

class AdaptivePerturbation:
    """
    An adaptive perturbation strategy for the randomized black box optimization algorithm.
    
    The strategy adjusts the perturbation amount based on the fitness of the current solution.
    This leads to a more adaptive and effective search process.
    """
    
    def __init__(self, budget, dim):
        """
        Initialize the perturbation strategy with a budget and dimensionality.
        
        Args:
            budget (int): The maximum number of function evaluations allowed.
            dim (int): The dimensionality of the search space.
        """
        self.budget = budget
        self.dim = dim
        self.search_space = [(-5.0, 5.0)] * dim
        self.fitness = {}
        self.perturbation_amount = 1.0
        self.perturbation_history = []
    
    def __call__(self, func):
        """
        Optimize the black box function using the perturbation strategy.
        
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
            
            # Update the fitness of the solution
            self.fitness[perturbed_solution] = new_cost
            
            # Store the perturbation amount in the history
            self.perturbation_amount = random.uniform(0.7, 0.9)
            self.perturbation_history.append((perturbed_solution, new_cost))
        
        return solution, cost
    
    def perturb(self, solution):
        """
        Perturb the current solution with an adaptive amount.
        
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
        Run the perturbation strategy for a specified number of iterations.
        
        Args:
            func (function): The black box function to optimize.
            num_iterations (int): The number of iterations to run.
        
        Returns:
            tuple: The optimal solution and its cost.
        """
        
        # Run the perturbation strategy for the specified number of iterations
        for _ in range(num_iterations):
            solution, cost = self(func)
            self.func_evaluations += 1
            
            # If the optimizer has reached the budget, break the loop
            if self.func_evaluations >= self.budget:
                break
            
            # Update the fitness of the solution
            self.fitness[solution] = cost
            
            # Store the perturbation amount in the history
            self.perturbation_amount = random.uniform(0.7, 0.9)
            self.perturbation_history.append((solution, cost))
        
        # Refine the strategy based on the fitness history
        refined_perturbation_amount = 0.8
        refined_perturbation_history = []
        for perturbed_solution, new_cost in self.perturbation_history:
            refined_perturbation_amount *= 0.9
            refined_perturbation_history.append((perturbed_solution, new_cost))
        
        # Update the perturbation strategy
        self.perturbation_amount = refined_perturbation_amount
        self.perturbation_history = refined_perturbation_history
    
    def evaluate_fitness(self, individual):
        """
        Evaluate the fitness of an individual using the perturbation strategy.
        
        Args:
            individual (tuple): The individual to evaluate.
        
        Returns:
            float: The fitness of the individual.
        """
        
        # Initialize the fitness
        fitness = 0
        
        # Perturb the individual
        for _ in range(self.budget):
            # Perturb the individual
            perturbed_individual = self.perturb(individual)
            
            # Evaluate the new individual
            fitness += self.func(perturbed_individual)
        
        # Return the fitness
        return fitness
    
# Description: Adaptive Perturbation for Enhanced Optimization
# Code: 