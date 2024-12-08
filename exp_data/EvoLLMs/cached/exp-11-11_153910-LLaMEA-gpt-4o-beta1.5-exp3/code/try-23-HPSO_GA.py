import numpy as np

class HPSO_GA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 30
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population = np.random.uniform(self.lower_bound, self.upper_bound, (self.pop_size, self.dim))
        self.velocities = np.random.uniform(-1, 1, (self.pop_size, self.dim))
        self.fitness = np.full(self.pop_size, float('inf'))
        self.pbest_positions = self.population.copy()
        self.pbest_fitness = self.fitness.copy()
        self.gbest_position = None
        self.gbest_fitness = float('inf')
        self.evaluations = 0
        self.w = 0.5  # Inertia weight
        self.c1 = 1.5 # Cognitive component
        self.c2 = 1.5 # Social component

    def evaluate(self, func, solution):
        self.evaluations += 1
        return func(solution)

    def update_particles(self, func):
        for i in range(self.pop_size):
            self.velocities[i] = (self.w * self.velocities[i] +
                                  self.c1 * np.random.rand() * (self.pbest_positions[i] - self.population[i]) +
                                  self.c2 * np.random.rand() * (self.gbest_position - self.population[i]))
            self.population[i] += self.velocities[i]
            self.population[i] = np.clip(self.population[i], self.lower_bound, self.upper_bound)
            fitness_i = self.evaluate(func, self.population[i])
            if fitness_i < self.pbest_fitness[i]:
                self.pbest_fitness[i] = fitness_i
                self.pbest_positions[i] = self.population[i]
            if fitness_i < self.gbest_fitness:
                self.gbest_fitness = fitness_i
                self.gbest_position = self.population[i]

    def genetic_crossover(self, parent1, parent2):
        alpha = np.random.rand(self.dim)
        child1 = alpha * parent1 + (1 - alpha) * parent2
        child2 = alpha * parent2 + (1 - alpha) * parent1
        return child1, child2

    def genetic_mutation(self, offspring):
        mutation_rate = 1.0 / self.dim
        mutation_vector = np.random.rand(self.dim) < mutation_rate
        mutation_values = np.random.uniform(self.lower_bound, self.upper_bound, self.dim)
        return np.where(mutation_vector, mutation_values, offspring)

    def genetic_operations(self, func):
        for i in range(0, self.pop_size, 2):
            if self.evaluations >= self.budget:
                break
            parent1, parent2 = self.population[np.random.choice(self.pop_size, 2, replace=False)]
            child1, child2 = self.genetic_crossover(parent1, parent2)
            child1 = self.genetic_mutation(child1)
            child2 = self.genetic_mutation(child2)
            child1 = np.clip(child1, self.lower_bound, self.upper_bound)
            child2 = np.clip(child2, self.lower_bound, self.upper_bound)
            for child in [child1, child2]:
                if self.evaluations < self.budget:
                    fitness_child = self.evaluate(func, child)
                    if fitness_child < np.max(self.fitness):
                        worst_idx = np.argmax(self.fitness)
                        self.population[worst_idx] = child
                        self.fitness[worst_idx] = fitness_child
                        if fitness_child < self.gbest_fitness:
                            self.gbest_fitness = fitness_child
                            self.gbest_position = child

    def __call__(self, func):
        np.random.seed(42)
        
        # Initial evaluation
        for i in range(self.pop_size):
            self.fitness[i] = self.evaluate(func, self.population[i])
            self.pbest_fitness[i] = self.fitness[i]
            if self.fitness[i] < self.gbest_fitness:
                self.gbest_fitness = self.fitness[i]
                self.gbest_position = self.population[i]

        while self.evaluations < self.budget:
            self.update_particles(func)
            self.genetic_operations(func)

        return self.gbest_position