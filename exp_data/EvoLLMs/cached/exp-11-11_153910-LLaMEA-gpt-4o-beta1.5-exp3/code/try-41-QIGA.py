import numpy as np

class QIGA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 30
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population = np.random.uniform(self.lower_bound, self.upper_bound, (self.pop_size, self.dim))
        self.fitness = np.full(self.pop_size, float('inf'))
        self.best_global_position = None
        self.best_global_fitness = float('inf')
        self.evaluations = 0
        self.q_bits = np.random.uniform(0, 1, (self.pop_size, self.dim))  # Quantum bits representing probabilities

    def evaluate(self, func, solution):
        self.evaluations += 1
        return func(solution)

    def quantum_update(self, idx):
        theta = np.arccos(self.q_bits[idx])  # Map q-bits to angles
        delta_theta = np.random.uniform(-0.1, 0.1, self.dim)  # Small variation
        theta += delta_theta
        self.q_bits[idx] = np.cos(theta)  # Update q-bits
        self.population[idx] = self.lower_bound + (self.upper_bound - self.lower_bound) * self.q_bits[idx]  # Map q-bits to solutions

    def crossover(self, parent1, parent2):
        crossover_point = np.random.randint(1, self.dim - 1)
        child1 = np.concatenate((parent1[:crossover_point], parent2[crossover_point:]))
        child2 = np.concatenate((parent2[:crossover_point], parent1[crossover_point:]))
        return child1, child2

    def mutate(self, individual):
        mutation_point = np.random.randint(self.dim)
        individual[mutation_point] = np.random.uniform(self.lower_bound, self.upper_bound)
        return individual

    def __call__(self, func):
        np.random.seed(42)

        # Initial evaluation
        for i in range(self.pop_size):
            self.fitness[i] = self.evaluate(func, self.population[i])
            if self.fitness[i] < self.best_global_fitness:
                self.best_global_fitness = self.fitness[i]
                self.best_global_position = self.population[i]

        while self.evaluations < self.budget:
            # Quantum bit update
            for i in range(self.pop_size):
                self.quantum_update(i)

            # Evaluate new population
            for i in range(self.pop_size):
                self.fitness[i] = self.evaluate(func, self.population[i])
                if self.fitness[i] < self.best_global_fitness:
                    self.best_global_fitness = self.fitness[i]
                    self.best_global_position = self.population[i]

            # Selection
            selected_indices = np.argsort(self.fitness)[:self.pop_size//2]
            selected_population = self.population[selected_indices]

            # Crossover and Mutation
            new_population = []
            while len(new_population) < self.pop_size:
                parents = np.random.choice(selected_population.shape[0], 2, replace=False)
                child1, child2 = self.crossover(selected_population[parents[0]], selected_population[parents[1]])
                new_population.append(self.mutate(child1))
                if len(new_population) < self.pop_size:
                    new_population.append(self.mutate(child2))
            
            self.population = np.array(new_population)

        return self.best_global_position