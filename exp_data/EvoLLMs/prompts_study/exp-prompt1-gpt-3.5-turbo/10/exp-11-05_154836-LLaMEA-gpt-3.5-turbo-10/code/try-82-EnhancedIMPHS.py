import numpy as np

class EnhancedIMPHS(IMPHS):
    def exploit_phase(self, population, num_iterations=5):
        for _ in range(num_iterations):
            best_idx = np.argmin(self.evaluate_population(population))
            best_individual = population[best_idx]
            new_population = population + np.random.uniform(-0.1, 0.1, population.shape)
            new_population[best_idx] = best_individual
            new_population = np.clip(new_population, self.lower_bound, self.upper_bound)

            trial_population = population + 0.5 * (population - new_population[np.random.choice(len(new_population))])
            trial_population = np.clip(trial_population, self.lower_bound, self.upper_bound)
            trial_evaluations = self.evaluate_population(trial_population)

            population[trial_evaluations < self.evaluate_population(population)] = trial_population[trial_evaluations < self.evaluate_population(population)]

        return population