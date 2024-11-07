import numpy as np

class EnhancedIMPHS(IMPHS):
    def __init__(self, budget, dim):
        super().__init__(budget, dim)

    def exploit_phase(population, num_iterations=5):
        for _ in range(num_iterations):
            scores = evaluate_population(population)
            best_idx = np.argmin(scores)
            best_individual = population[best_idx]
            new_population = population + np.random.uniform(-0.1, 0.1, population.shape)
            new_population[best_idx] = best_individual
            new_population = np.clip(new_population, self.lower_bound, self.upper_bound)
            population = new_population

            # Dynamic adaptation mechanism
            for i in range(len(population)):
                if np.random.rand() < 0.1:
                    population[i] = population[i] + np.random.uniform(-0.1, 0.1, population.shape[i])

        return population