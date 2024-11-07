import numpy as np

class EnhancedAdaptiveMutationGeneticAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10
        self.initial_mutation_prob = 0.1
        self.initial_crossover_rate = 0.9

    def __call__(self, func):
        population = np.random.uniform(-5.0, 5.0, (self.population_size, self.dim))
        mutation_prob = np.full(self.population_size, self.initial_mutation_prob)
        crossover_rate = np.full(self.population_size, self.initial_crossover_rate)

        for _ in range(self.budget):
            fitness = np.array([func(individual) for individual in population])
            fitness_sorted_idx = np.argsort(fitness)
            fittest_individual = population[fitness_sorted_idx[0]]

            new_population = [fittest_individual]
            for i in range(1, self.population_size):
                parents_idx = np.random.choice(self.population_size, 3, replace=False)
                parent1, parent2, parent3 = population[parents_idx]
                if i == 1:  # Elitism - Keep the best individual
                    child = fittest_individual + mutation_prob[i] * np.random.uniform(-1, 1, self.dim)
                else:
                    child = parent1 + mutation_prob[i] * (parent2 - parent3)

                if np.random.uniform() < mutation_prob[i]:
                    child += np.random.normal(0, 1, self.dim) * (1 - 0.99 * (_ / self.budget))

                if np.random.uniform() < crossover_rate[i]:
                    cross_points = np.random.randint(0, 2, self.dim).astype(bool)
                    child[cross_points] = parent1[cross_points]

                new_population.append(child)

                # Update mutation and crossover rates based on individual performance
                if i > 1:
                    mutation_prob[i] = np.clip(mutation_prob[i] + np.random.normal(0, 0.1), 0.01, 0.5)
                    crossover_rate[i] = np.clip(crossover_rate[i] + np.random.normal(0, 0.1), 0.7, 0.99)

            population = np.array(new_population)

        best_idx = np.argmin([func(individual) for individual in population])
        return population[best_idx]