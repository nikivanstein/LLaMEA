import numpy as np

class EnhancedDynamicMultiStrategyEA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.strategy_count = 3
        self.population_size = 20
        self.init_strategy_weights()

    def init_strategy_weights(self):
        self.strategy_weights = np.full(self.strategy_count, 1.0 / self.strategy_count)

    def select_strategy(self):
        return np.random.choice(self.strategy_count, p=self.strategy_weights)

    def mutate(self, x, strategy):
        if strategy == 0:
            return x + np.random.normal(0, 0.1, size=self.dim)  # Adjusted mutation
        elif strategy == 1:
            return x + np.random.uniform(-0.1, 0.1, size=self.dim)  # Adjusted mutation
        else:
            scale = np.random.uniform(0.7, 1.3)  # Adjusted scale range
            return x * scale

    def boundary_check(self, x):
        return np.clip(x, self.lower_bound, self.upper_bound)

    def adapt_strategy_weights(self, success_counts):
        total = np.sum(success_counts)
        if total > 0:
            self.strategy_weights = 0.9 * (success_counts / total) + 0.1 * self.strategy_weights
        else:
            self.init_strategy_weights()

    def __call__(self, func):
        population_size = min(self.population_size, self.budget // 5)  # Adaptive population size
        population = np.random.uniform(self.lower_bound, self.upper_bound, (population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        evaluations = population_size
        
        success_counts = np.zeros(self.strategy_count)

        while evaluations < self.budget:
            offspring = []
            offspring_fitness = []
            for _ in range(population_size):
                parent_index = np.random.choice(population_size)
                parent = population[parent_index]
                strategy = self.select_strategy()
                child = self.mutate(parent, strategy)
                child = self.boundary_check(child)
                child_fitness = func(child)
                evaluations += 1
                if child_fitness < fitness[parent_index]:
                    offspring.append(child)
                    offspring_fitness.append(child_fitness)
                    success_counts[strategy] += 1
                if evaluations >= self.budget:
                    break

            if offspring:
                population = np.concatenate((population, np.array(offspring)), axis=0)
                fitness = np.concatenate((fitness, np.array(offspring_fitness)), axis=0)
                best_indices = fitness.argsort()[:population_size]
                population = population[best_indices]
                fitness = fitness[best_indices]

            self.adapt_strategy_weights(success_counts)
            success_counts.fill(0)