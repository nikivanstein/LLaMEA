import numpy as np

class DualLayeredMemeticAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = int(10 + 2 * np.sqrt(dim))
        self.population = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, dim))
        self.func_evaluations = 0
        self.best_score = float('inf')
        self.best_position = None
        self.mutation_rate = 0.1
        self.crossover_rate = 0.8

    def __call__(self, func):
        while self.func_evaluations < self.budget:
            # Evaluate fitness
            fitness = np.array([func(individual) for individual in self.population])
            self.func_evaluations += self.population_size

            # Update best individual
            min_index = np.argmin(fitness)
            if fitness[min_index] < self.best_score:
                self.best_score = fitness[min_index]
                self.best_position = self.population[min_index].copy()

            # Selection
            selected_indices = np.random.choice(self.population_size, self.population_size, p=fitness/fitness.sum())
            selected_population = self.population[selected_indices]

            # Crossover
            next_population = []
            for i in range(0, self.population_size, 2):
                if np.random.rand() < self.crossover_rate:
                    point = np.random.randint(1, self.dim)
                    parent1, parent2 = selected_population[i], selected_population[i+1]
                    child1 = np.concatenate((parent1[:point], parent2[point:]))
                    child2 = np.concatenate((parent2[:point], parent1[point:]))
                    next_population.extend([child1, child2])
                else:
                    next_population.extend([selected_population[i], selected_population[i+1]])

            # Mutation
            next_population = np.array(next_population)
            mutation_array = np.random.rand(*next_population.shape) < self.mutation_rate
            random_mutations = np.random.uniform(self.lower_bound, self.upper_bound, next_population.shape)
            next_population = np.where(mutation_array, random_mutations, next_population)
            self.population = np.clip(next_population, self.lower_bound, self.upper_bound)

            # Local Search
            for i in range(self.population_size):
                if self.func_evaluations >= self.budget:
                    break
                local_best = self.local_search(self.population[i], func)
                local_score = func(local_best)
                self.func_evaluations += 1
                if local_score < fitness[i]:
                    self.population[i] = local_best

        return self.best_position

    def local_search(self, position, func):
        step_size = 0.1 * (self.upper_bound - self.lower_bound)
        for _ in range(5):  # Perform 5 local search steps
            for d in range(self.dim):
                new_pos = position.copy()
                new_pos[d] += np.random.uniform(-step_size, step_size)
                new_pos[d] = np.clip(new_pos[d], self.lower_bound, self.upper_bound)
                if func(new_pos) < func(position):
                    position = new_pos
        return position