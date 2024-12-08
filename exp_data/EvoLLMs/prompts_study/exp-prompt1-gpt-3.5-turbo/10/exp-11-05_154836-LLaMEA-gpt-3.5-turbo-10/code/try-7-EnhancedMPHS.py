import numpy as np

class EnhancedMPHS(MPHS):
    def __call__(self, func):
        def evaluate_population(population):
            fitness_values = np.array([func(individual) for individual in population])
            fitness_order = np.argsort(fitness_values)
            return population[fitness_order], fitness_values

        def adapt_phase(population, fitness_values):
            diversity = np.std(population, axis=0)
            exploration_prob = np.clip(0.1 * np.mean(diversity), 0.1, 0.9)
            exploitation_prob = 1 - exploration_prob
            return exploration_prob, exploitation_prob

        population = initialize_population()
        for _ in range(self.budget // 2):
            sorted_population, sorted_fitness = evaluate_population(population)
            exploration_prob, exploitation_prob = adapt_phase(sorted_population, sorted_fitness)
            if np.random.rand() < exploration_prob:
                population = explore_phase(sorted_population)
            else:
                population = exploit_phase(sorted_population)
        best_individual = sorted_population[np.argmin(sorted_fitness)]
        return best_individual