import numpy as np
import random

class HybridMetaheuristic:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 50
        self.memory = []
        self.pbest = np.zeros((self.population_size, self.dim))
        self.gbest = np.zeros(self.dim)
        self.pbest_count = np.zeros((self.population_size, self.dim))
        self.random_solution = np.random.uniform(-5.0, 5.0, size=self.dim)

    def __call__(self, func):
        for _ in range(self.budget):
            # Generate a new population
            population = np.random.uniform(-5.0, 5.0, size=(self.population_size, self.dim))
            # Evaluate the population
            evaluations = func(population)
            # Update the population with the best solutions
            for i in range(self.population_size):
                if evaluations[i] < evaluations[self.pbest_count[i, :]]:
                    self.pbest[i, :] = population[i, :]
                    self.pbest_count[i, :] = i
            # Update the global best solution
            min_evaluation = np.min(evaluations)
            if min_evaluation < self.gbest[np.argmin(self.gbest)]:
                self.gbest = evaluations[np.argmin(self.gbest)]
            # Update the memory with the best solutions
            self.memory.append(self.pbest[self.pbest_count == i, :])
            # Apply memetic operators
            for i in range(self.population_size):
                # Select a random solution from the memory
                random_solution = np.random.choice(self.memory, size=1)[0]
                # Mutate the solution
                mutation = np.random.uniform(-0.1, 0.1, size=self.dim)
                mutated_solution = random_solution + mutation
                # Apply crossover
                crossover = np.random.choice([0, 1], size=self.dim)
                if crossover[0] == 1:
                    mutated_solution[crossover[1]] = random_solution[crossover[1]]
                # Replace the solution with the mutated solution
                population[i, :] = mutated_solution
            # Evaluate the population again
            evaluations = func(population)
            # Update the population with the best solutions
            for i in range(self.population_size):
                if evaluations[i] < evaluations[self.pbest_count[i, :]]:
                    self.pbest[i, :] = population[i, :]
                    self.pbest_count[i, :] = i
            # Update the global best solution
            min_evaluation = np.min(evaluations)
            if min_evaluation < self.gbest[np.argmin(self.gbest)]:
                self.gbest = evaluations[np.argmin(self.gbest)]
        return self.gbest

    def select_random_solution(self):
        # Select a random solution from the population
        random_solution = np.random.choice(self.pbest, size=1)[0]
        return random_solution

    def apply_crossover(self, parent1, parent2):
        # Apply crossover
        crossover = np.random.choice([0, 1], size=self.dim)
        if crossover[0] == 1:
            child = parent1.copy()
            child[crossover[1]] = parent2[crossover[1]]
        else:
            child = parent2.copy()
            child[crossover[0]] = parent1[crossover[0]]
        return child

    def apply_mutation(self, child):
        # Apply mutation
        mutation = np.random.uniform(-0.1, 0.1, size=self.dim)
        child += mutation
        return child

    def update_population(self, population):
        # Update the population
        new_population = population.copy()
        for i in range(self.population_size):
            # Select a random solution from the population
            random_solution = self.select_random_solution()
            # Apply crossover
            child = self.apply_crossover(random_solution, new_population[i])
            # Apply mutation
            child = self.apply_mutation(child)
            # Replace the solution with the child
            new_population[i, :] = child
        return new_population

# Refine the strategy by adding a new operator to the memetic operators
class HybridMetaheuristicRefined(HybridMetaheuristic):
    def __init__(self, budget, dim):
        super().__init__(budget, dim)
        self.operator_count = 4

    def __call__(self, func):
        for _ in range(self.budget):
            # Generate a new population
            population = np.random.uniform(-5.0, 5.0, size=(self.population_size, self.dim))
            # Evaluate the population
            evaluations = func(population)
            # Update the population with the best solutions
            for i in range(self.population_size):
                if evaluations[i] < evaluations[self.pbest_count[i, :]]:
                    self.pbest[i, :] = population[i, :]
                    self.pbest_count[i, :] = i
            # Update the global best solution
            min_evaluation = np.min(evaluations)
            if min_evaluation < self.gbest[np.argmin(self.gbest)]:
                self.gbest = evaluations[np.argmin(self.gbest)]
            # Update the memory with the best solutions
            self.memory.append(self.pbest[self.pbest_count == i, :])
            # Apply memetic operators
            for i in range(self.population_size):
                # Select a random solution from the memory
                random_solution = np.random.choice(self.memory, size=1)[0]
                # Mutate the solution
                mutation = np.random.uniform(-0.1, 0.1, size=self.dim)
                mutated_solution = random_solution + mutation
                # Apply crossover
                crossover = np.random.choice([0, 1], size=self.dim)
                if crossover[0] == 1:
                    mutated_solution[crossover[1]] = random_solution[crossover[1]]
                # Apply mutation
                mutated_solution = self.apply_mutation(mutated_solution)
                # Apply new operator
                if np.random.rand() < 0.2:
                    new_operator = np.random.choice([0, 1, 2, 3])
                    if new_operator == 0:
                        # Select a random solution from the population
                        parent1 = np.random.choice(self.pbest, size=1)[0]
                        # Select a random solution from the population
                        parent2 = np.random.choice(self.pbest, size=1)[0]
                        # Apply crossover
                        child = self.apply_crossover(parent1, parent2)
                        # Apply mutation
                        child = self.apply_mutation(child)
                    elif new_operator == 1:
                        # Select a random solution from the memory
                        parent1 = np.random.choice(self.memory, size=1)[0]
                        # Select a random solution from the population
                        parent2 = np.random.choice(self.pbest, size=1)[0]
                        # Apply crossover
                        child = self.apply_crossover(parent1, parent2)
                        # Apply mutation
                        child = self.apply_mutation(child)
                    elif new_operator == 2:
                        # Select a random solution from the population
                        parent1 = np.random.choice(self.pbest, size=1)[0]
                        # Select a random solution from the population
                        parent2 = np.random.choice(self.pbest, size=1)[0]
                        # Apply crossover
                        child = self.apply_crossover(parent1, parent2)
                        # Apply mutation
                        child = self.apply_mutation(child)
                    else:
                        # Select a random solution from the memory
                        parent1 = np.random.choice(self.memory, size=1)[0]
                        # Select a random solution from the population
                        parent2 = np.random.choice(self.pbest, size=1)[0]
                        # Apply crossover
                        child = self.apply_crossover(parent1, parent2)
                        # Apply mutation
                        child = self.apply_mutation(child)
                    # Replace the solution with the child
                    population[i, :] = child
            # Evaluate the population again
            evaluations = func(population)
            # Update the population with the best solutions
            for i in range(self.population_size):
                if evaluations[i] < evaluations[self.pbest_count[i, :]]:
                    self.pbest[i, :] = population[i, :]
                    self.pbest_count[i, :] = i
            # Update the global best solution
            min_evaluation = np.min(evaluations)
            if min_evaluation < self.gbest[np.argmin(self.gbest)]:
                self.gbest = evaluations[np.argmin(self.gbest)]
        return self.gbest