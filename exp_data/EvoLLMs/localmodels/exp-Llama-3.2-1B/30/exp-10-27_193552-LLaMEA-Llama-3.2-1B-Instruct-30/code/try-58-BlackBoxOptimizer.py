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
        
        # Initialize the population with random solutions
        population = self.generate_population(func, num_iterations)
        
        # Run the optimizer for the specified number of iterations
        for _ in range(num_iterations):
            # Select the fittest individual
            fittest_individual = population[np.argmax([self.evaluate_fitness(individual) for individual in population])]
            
            # Select two parents using tournament selection
            parents = [self.select_parents(fittest_individual, population, num_iterations)]
            
            # Crossover (recombination) the parents to create offspring
            offspring = self.crossover(parents)
            
            # Mutate the offspring to introduce genetic variation
            offspring = self.mutate(offspring)
            
            # Evaluate the new population
            new_population = self.evaluate_population(offspring, func, num_iterations)
            
            # Replace the old population with the new one
            population = new_population
        
        # Return the fittest individual from the final population
        return self.select_fittest(population, func, num_iterations)
    
    def generate_population(self, func, num_individuals):
        """
        Generate a population of random solutions.
        
        Args:
            func (function): The black box function to optimize.
            num_individuals (int): The number of individuals in the population.
        
        Returns:
            list: The population of random solutions.
        """
        
        population = []
        for _ in range(num_individuals):
            solution = func(np.random.uniform(-5.0, 5.0, self.dim))
            population.append(solution)
        
        return population
    
    def select_parents(self, fittest_individual, population, num_parents):
        """
        Select two parents using tournament selection.
        
        Args:
            fittest_individual (tuple): The fittest individual.
            population (list): The population of random solutions.
            num_parents (int): The number of parents to select.
        
        Returns:
            list: The selected parents.
        """
        
        parents = []
        for _ in range(num_parents):
            tournament_size = random.randint(1, len(population))
            tournament = random.sample(population, tournament_size)
            winner = np.argmax([self.evaluate_fitness(individual) for individual in tournament])
            parents.append((tournament[winner], tournament[winner]))
        
        return parents
    
    def crossover(self, parents):
        """
        Crossover (recombination) the parents to create offspring.
        
        Args:
            parents (list): The parents to crossover.
        
        Returns:
            list: The offspring.
        """
        
        offspring = []
        for _ in range(len(parents)):
            parent1, parent2 = parents[_]
            offspring.append((parent1[0] + parent2[0]) / 2, (parent1[1] + parent2[1]) / 2)
        
        return offspring
    
    def mutate(self, offspring):
        """
        Mutate the offspring to introduce genetic variation.
        
        Args:
            offspring (list): The offspring to mutate.
        
        Returns:
            list: The mutated offspring.
        """
        
        mutated_offspring = []
        for individual in offspring:
            mutated_individual = (individual[0] + random.uniform(-1, 1), individual[1] + random.uniform(-1, 1))
            mutated_offspring.append(mutated_individual)
        
        return mutated_offspring
    
    def evaluate_fitness(self, individual):
        """
        Evaluate the fitness of an individual.
        
        Args:
            individual (tuple): The individual to evaluate.
        
        Returns:
            float: The fitness of the individual.
        """
        
        # Evaluate the function at the individual
        func_value = func(individual)
        
        # Return the fitness
        return func_value
    
    def evaluate_population(self, offspring, func, num_evaluations):
        """
        Evaluate the fitness of a population of offspring.
        
        Args:
            offspring (list): The offspring to evaluate.
            func (function): The black box function to optimize.
            num_evaluations (int): The number of function evaluations.
        
        Returns:
            float: The average fitness of the offspring.
        """
        
        # Initialize the total fitness
        total_fitness = 0
        
        # Evaluate the fitness of each offspring
        for individual in offspring:
            func_value = func(individual)
            total_fitness += func_value
        
        # Return the average fitness
        return total_fitness / num_evaluations
    
    def select_fittest(self, population, func, num_evaluations):
        """
        Select the fittest individual from a population.
        
        Args:
            population (list): The population of random solutions.
            func (function): The black box function to optimize.
            num_evaluations (int): The number of function evaluations.
        
        Returns:
            tuple: The fittest individual.
        """
        
        # Initialize the fittest individual
        fittest_individual = population[0]
        
        # Initialize the minimum fitness
        min_fitness = self.evaluate_fitness(fittest_individual)
        
        # Evaluate the fitness of each individual
        for individual in population:
            func_value = func(individual)
            if func_value < min_fitness:
                fittest_individual = individual
                min_fitness = func_value
        
        return fittest_individual
    
    def evaluateBBOB(self, func, num_iterations):
        """
        Evaluate the fitness of the BBOB test suite.
        
        Args:
            func (function): The black box function to optimize.
            num_iterations (int): The number of iterations to run.
        
        Returns:
            tuple: The optimal solution and its cost.
        """
        
        # Initialize the population with random solutions
        population = self.generate_population(func, 100)
        
        # Run the optimizer for the specified number of iterations
        for _ in range(num_iterations):
            # Select the fittest individual
            fittest_individual = self.select_fittest(population, func, 100)
            
            # Select two parents using tournament selection
            parents = [self.select_parents(fittest_individual, population, 100)]
            
            # Crossover (recombination) the parents to create offspring
            offspring = self.crossover(parents)
            
            # Mutate the offspring to introduce genetic variation
            offspring = self.mutate(offspring)
            
            # Evaluate the new population
            new_population = self.evaluate_population(offspring, func, 100)
            
            # Replace the old population with the new one
            population = new_population
        
        # Return the fittest individual from the final population
        return self.select_fittest(population, func, 100)