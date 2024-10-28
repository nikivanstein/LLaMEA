import random
import numpy as np

class GeneticOptimizer:
    """
    An optimization algorithm that uses genetic algorithm to find the optimal solution.
    
    Parameters:
    budget (int): The maximum number of function evaluations allowed.
    dim (int): The dimensionality of the search space.
    """
    
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 100
        self.mutation_rate = 0.1
        self.population = []
        self.fitness_scores = []

    def __call__(self, func):
        """
        Optimize the black box function using the given budget for function evaluations.
        
        Parameters:
        func (function): The black box function to optimize.
        
        Returns:
        tuple: A tuple containing the optimal solution and its cost.
        """
        
        # Initialize the search space
        lower_bound = -5.0
        upper_bound = 5.0
        
        # Initialize the best solution and its cost
        best_solution = None
        best_cost = float('inf')
        
        # Generate an initial population of random solutions
        for _ in range(self.population_size):
            individual = (lower_bound + random.uniform(-5.0, 5.0)) / 2
            fitness = self.f(individual, func)
            self.population.append(individual)
            self.fitness_scores.append(fitness)
        
        # Evolve the population using crossover and mutation
        while self.population_size > 0:
            # Select parents using tournament selection
            parents = self.tournament_selection(self.population)
            
            # Crossover to create offspring
            offspring = self.crossover(parents)
            
            # Mutate the offspring
            offspring = self.mutate(offspring)
            
            # Replace the least fit individual with the new offspring
            self.population[self.population_size - 1] = offspring[0]
            self.population_size -= 1
            
            # Update the best solution and its cost
            if self.fitness_scores[-1] < best_cost:
                best_solution = self.population[0]
                best_cost = self.fitness_scores[-1]
        
        # Return the optimal solution and its cost
        return best_solution, best_cost

    def tournament_selection(self, population):
        """
        Select parents using tournament selection.
        
        Parameters:
        population (list): A list of individuals.
        
        Returns:
        list: A list of selected parents.
        """
        
        # Select the top 3 parents with the highest fitness scores
        selected_parents = sorted(population, key=self.fitness_scores[-1], reverse=True)[:3]
        
        return selected_parents

    def crossover(self, parents):
        """
        Crossover to create offspring.
        
        Parameters:
        parents (list): A list of parents.
        
        Returns:
        list: A list of offspring.
        """
        
        # Select a random crossover point
        crossover_point = random.randint(0, len(parents) - 1)
        
        # Create the offspring
        offspring = parents[:crossover_point] + parents[crossover_point + 1:]
        
        return offspring

    def mutate(self, individual):
        """
        Mutate the individual.
        
        Parameters:
        individual (list): An individual.
        
        Returns:
        list: The mutated individual.
        """
        
        # Select a random mutation point
        mutation_point = random.randint(0, len(individual) - 1)
        
        # Flip the bit at the mutation point
        individual[mutation_point] = 1 - individual[mutation_point]
        
        return individual

# Description: Genetic Algorithm for BBOB Black Box Optimizer
# Code: 
# ```python
# import numpy as np
# import random
# import time
# 
# def black_box_optimizer(budget, dim):
#     optimizer = GeneticOptimizer(budget, dim)
#     func_evals = 0
#     best_solution = None
#     best_cost = float('inf')
# 
#     while True:
#         # Optimize the function using the optimizer
#         solution, cost = optimizer(func)
# 
#         # Increment the number of function evaluations
#         func_evals += 1
# 
#         # If the number of function evaluations exceeds the budget, break the loop
#         if func_evals > budget:
#             break
        
#         # Update the best solution and its cost
#         if cost < best_cost:
            # best_solution = solution
            # best_cost = cost
#     return best_solution, best_cost
# 
# def main():
#     budget = 1000
#     dim = 10
#     best_solution, best_cost = black_box_optimizer(budget, dim)
#     print("Optimal solution:", best_solution)
#     print("Optimal cost:", best_cost)
# 
# if __name__ == "__main__":
#     main()