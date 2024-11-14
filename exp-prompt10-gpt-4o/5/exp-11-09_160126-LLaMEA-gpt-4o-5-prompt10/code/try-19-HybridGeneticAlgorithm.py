import numpy as np
import random

class HybridGeneticAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 20
        self.mutation_rate = 0.1
        self.elite_fraction = 0.2
        self.tournament_size = 4
        self.lb = -5.0
        self.ub = 5.0
        self.evaluations = 0

    def _initialize_population(self):
        return np.random.uniform(self.lb, self.ub, (self.population_size, self.dim))
    
    def _evaluate_population(self, population, func):
        return np.array([func(ind) for ind in population])
    
    def _select_parents(self, fitness):
        selected = []
        for _ in range(self.population_size):
            tournament = np.random.choice(self.population_size, min(self.tournament_size, self.population_size), replace=False)
            best = tournament[np.argmin(fitness[tournament])]
            selected.append(best)
        return selected

    def _adaptive_crossover(self, parent1, parent2, iteration, max_iterations):
        crossover_point = np.random.randint(1, self.dim-1)
        if iteration < max_iterations // 2:
            child = np.concatenate((parent1[:crossover_point], parent2[crossover_point:]))
        else:
            alpha = np.random.uniform(0, 1, self.dim)
            child = alpha * parent1 + (1 - alpha) * parent2
        return child
    
    def _adaptive_mutate(self, individual, iteration, max_iterations):
        mutation_strength = np.interp(iteration, [0, max_iterations], [0.1, 0.01])
        for i in range(self.dim):
            if np.random.rand() < self.mutation_rate:
                individual[i] += np.random.normal(0, mutation_strength)
                individual[i] = np.clip(individual[i], self.lb, self.ub)
        return individual

    def _local_search(self, individual, func):
        neighbors = [individual + np.random.uniform(-0.1, 0.1, self.dim) for _ in range(5)]
        neighbors = np.clip(neighbors, self.lb, self.ub)
        neighbor_fitness = [func(neighbor) for neighbor in neighbors]
        best_neighbor = neighbors[np.argmin(neighbor_fitness)]
        return best_neighbor

    def __call__(self, func):
        # Initialize population
        population = self._initialize_population()
        fitness = self._evaluate_population(population, func)
        self.evaluations += self.population_size
        iteration = 0

        while self.evaluations < self.budget:
            # Select parents
            parents_idx = self._select_parents(fitness)
            new_population = []

            # Generate new population through adaptive crossover and mutation
            for i in range(0, self.population_size, 2):
                parent1 = population[parents_idx[i]]
                parent2 = population[parents_idx[i+1]]

                child1 = self._adaptive_crossover(parent1, parent2, iteration, self.budget // self.population_size)
                child2 = self._adaptive_crossover(parent2, parent1, iteration, self.budget // self.population_size)

                child1 = self._adaptive_mutate(child1, iteration, self.budget // self.population_size)
                child2 = self._adaptive_mutate(child2, iteration, self.budget // self.population_size)

                new_population.append(child1)
                new_population.append(child2)

            new_population = np.array(new_population)

            # Evaluate new population
            new_fitness = self._evaluate_population(new_population, func)
            self.evaluations += self.population_size
            iteration += 1

            # Combine old and new population and select elites
            combined_population = np.vstack((population, new_population))
            combined_fitness = np.hstack((fitness, new_fitness))
            elite_count = int(self.elite_fraction * self.population_size)

            elite_indices = np.argsort(combined_fitness)[:elite_count]
            population = combined_population[elite_indices]
            fitness = combined_fitness[elite_indices]

            # Apply local search on elites
            for i in range(elite_count):
                population[i] = self._local_search(population[i], func)
                fitness[i] = func(population[i])
                self.evaluations += 1
                if self.evaluations >= self.budget:
                    break
        
        # Return best found solution
        best_idx = np.argmin(fitness)
        return population[best_idx]