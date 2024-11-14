import numpy as np

class QuantumEntangledGA:
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
        self.entangled_points = np.zeros((2, dim))

    def __call__(self, func):
        while self.func_evaluations < self.budget:
            # Evaluate current population
            fitness = np.array([func(ind) for ind in self.population])
            self.func_evaluations += self.population_size

            # Track the best solution
            min_idx = np.argmin(fitness)
            if fitness[min_idx] < self.best_score:
                self.best_score = fitness[min_idx]
                self.best_position = self.population[min_idx].copy()

            # Select parents (tournament selection)
            selected_parents = self.tournament_selection(fitness)

            # Crossover (uniform)
            offspring = self.uniform_crossover(selected_parents)

            # Mutation (adaptive rate)
            self.adaptive_mutation(offspring)

            # Create new population
            self.population = offspring

            # Update entangled points for quantum-inspired mutation
            self.update_entangled_points()

        return self.best_position

    def tournament_selection(self, fitness):
        selected = []
        for _ in range(self.population_size):
            competitors = np.random.choice(self.population_size, 3, replace=False)
            winner = competitors[np.argmin(fitness[competitors])]
            selected.append(self.population[winner])
        return np.array(selected)

    def uniform_crossover(self, parents):
        offspring = np.empty_like(parents)
        for i in range(0, self.population_size, 2):
            if i+1 < self.population_size:
                mask = np.random.rand(self.dim) < 0.5
                offspring[i] = np.where(mask, parents[i], parents[i+1])
                offspring[i+1] = np.where(mask, parents[i+1], parents[i])
            else:
                offspring[i] = parents[i]
        return np.clip(offspring, self.lower_bound, self.upper_bound)

    def adaptive_mutation(self, offspring):
        mutation_probs = np.random.rand(self.population_size, self.dim)
        mutation_factors = np.random.normal(scale=self.mutation_rate, size=(self.population_size, self.dim))
        mutations = (mutation_probs < self.mutation_rate) * mutation_factors
        offspring += mutations
        np.clip(offspring, self.lower_bound, self.upper_bound, out=offspring)

        # Update mutation rate
        self.mutation_rate = 0.1 * (1 + np.cos(2 * np.pi * self.func_evaluations / self.budget))

    def update_entangled_points(self):
        sorted_idx = np.argsort([func(ind) for ind in self.population])
        best2 = self.population[sorted_idx[:2]]
        self.entangled_points = best2