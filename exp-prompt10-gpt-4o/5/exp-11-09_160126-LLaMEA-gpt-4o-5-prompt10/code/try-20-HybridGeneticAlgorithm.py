import numpy as np

class HybridGeneticAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.initial_population_size = 20
        self.mutation_rate = 0.1
        self.elite_fraction = 0.2
        self.tournament_size = 3
        self.lb = -5.0
        self.ub = 5.0
        self.evaluations = 0

    def _initialize_population(self, pop_size):
        return np.random.uniform(self.lb, self.ub, (pop_size, self.dim))
    
    def _evaluate_population(self, population, func):
        return np.array([func(ind) for ind in population])
    
    def _select_parents(self, fitness, pop_size):
        selected = []
        for _ in range(pop_size):
            tournament = np.random.choice(pop_size, self.tournament_size, replace=False)
            best = tournament[np.argmin(fitness[tournament])]
            selected.append(best)
        return selected

    def _crossover(self, parent1, parent2):
        crossover_point = np.random.randint(1, self.dim-1)
        child = np.concatenate((parent1[:crossover_point], parent2[crossover_point:]))
        return child
    
    def _mutate(self, individual, adaptive_rate):
        for i in range(self.dim):
            if np.random.rand() < adaptive_rate:
                individual[i] = np.random.uniform(self.lb, self.ub)
        return individual

    def _local_search(self, individual, func):
        neighbors = [individual + np.random.uniform(-0.05, 0.05, self.dim) for _ in range(3)]
        neighbors = np.clip(neighbors, self.lb, self.ub)
        neighbor_fitness = [func(neighbor) for neighbor in neighbors]
        best_neighbor = neighbors[np.argmin(neighbor_fitness)]
        return best_neighbor

    def __call__(self, func):
        # Initialize population
        population_size = self.initial_population_size
        population = self._initialize_population(population_size)
        fitness = self._evaluate_population(population, func)
        self.evaluations += population_size

        while self.evaluations < self.budget:
            # Select parents
            parents_idx = self._select_parents(fitness, population_size)
            new_population = []

            # Generate new population through crossover and mutation
            adaptive_mutation_rate = max(0.02, self.mutation_rate * (1 - self.evaluations / self.budget))
            for i in range(0, population_size, 2):
                parent1 = population[parents_idx[i % population_size]]
                parent2 = population[parents_idx[(i+1) % population_size]]

                child1 = self._crossover(parent1, parent2)
                child2 = self._crossover(parent2, parent1)

                child1 = self._mutate(child1, adaptive_mutation_rate)
                child2 = self._mutate(child2, adaptive_mutation_rate)

                new_population.append(child1)
                new_population.append(child2)
            
            new_population = np.array(new_population)

            # Evaluate new population
            new_fitness = self._evaluate_population(new_population, func)
            self.evaluations += population_size

            # Combine old and new population and select elites
            combined_population = np.vstack((population, new_population))
            combined_fitness = np.hstack((fitness, new_fitness))
            elite_count = int(self.elite_fraction * population_size)

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

            # Dynamically adjust population size for diversity
            population_size = min(self.initial_population_size + (self.evaluations // 100), 40)

        # Return best found solution
        best_idx = np.argmin(fitness)
        return population[best_idx]