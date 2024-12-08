import numpy as np
import random

class BBOBMetaheuristic:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.func_evals = 0

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

    def mutate(self, sol):
        # Refine the solution by changing a random line
        for i in range(self.dim):
            if random.random() < 0.25:
                sol[i] = random.uniform(-5.0, 5.0)
        
        # Return the mutated solution
        return sol

    def evaluate_fitness(self, sol):
        # Evaluate the function at the solution
        func_sol = self.__call__(func, sol)
        
        # Return the fitness (lower is better)
        return func_sol

class GeneticBBOBMetaheuristic(BBOBMetaheuristic):
    def __init__(self, budget, dim):
        super().__init__(budget, dim)
        self.population_size = 100
        self.population = []

    def create_individual(self):
        # Create a new individual with random lines
        return np.random.uniform(-5.0, 5.0, self.dim)

    def mutate_individual(self, individual):
        # Mutate the individual by changing a random line
        for i in range(self.dim):
            if random.random() < 0.25:
                individual[i] = random.uniform(-5.0, 5.0)
        
        # Return the mutated individual
        return individual

    def select_parents(self, num_parents):
        # Select parents from the population using tournament selection
        parents = []
        for _ in range(num_parents):
            tournament_size = random.randint(2, self.population_size)
            tournament = np.random.choice(self.population, size=tournament_size, replace=False)
            winner = np.argmax(self.evaluate_fitness(tournament))
            parents.append((tournament[winner], winner))
        
        # Return the selected parents
        return parents

    def breed_parents(self, parents):
        # Breed the parents to create offspring
        offspring = []
        while len(offspring) < self.population_size:
            parent1, parent2 = random.sample(parents, 2)
            offspring.append(self.mutate_individual(parent1) + self.mutate_individual(parent2))
        
        # Return the offspring
        return offspring

    def evolve(self, num_generations):
        # Evolve the population using selection, breeding, and mutation
        for _ in range(num_generations):
            parents = self.select_parents(self.population_size)
            offspring = self.breed_parents(parents)
            self.population = self.breed_offspring(offspring)

    def breed_offspring(self, offspring):
        # Breed the offspring to create new offspring
        new_offspring = []
        while len(new_offspring) < self.population_size:
            parent1, parent2 = random.sample(offspring, 2)
            new_offspring.append(self.mutate_individual(parent1) + self.mutate_individual(parent2))
        
        # Return the new offspring
        return new_offspring

# Description: Evolutionary Algorithm for Black Box Optimization using Genetic Programming
# Code: 