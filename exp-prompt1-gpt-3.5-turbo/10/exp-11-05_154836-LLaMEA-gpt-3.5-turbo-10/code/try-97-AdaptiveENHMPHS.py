import numpy as np

class AdaptiveENHMPHS(ENHMPHS):
    def explore_phase(population):
        mutation_rates = np.random.uniform(0.05, 0.2, self.dim)
        fitness_values = evaluate_population(population)
        avg_fitness = np.mean(fitness_values)
        std_fitness = np.std(fitness_values)
        adaptive_rates = np.where(fitness_values < avg_fitness, mutation_rates * np.exp(-0.1 * (avg_fitness - fitness_values) / std_fitness), mutation_rates)
        mutation_strategies = [np.random.choice([np.random.normal, np.random.uniform], p=[0.7, 0.3]) for _ in range(self.dim)]
        new_population = population.copy()
        for i in range(self.dim):
            new_population[:, i] = mutation_strategies[i](population[:, i], adaptive_rates[i])
        return np.clip(new_population, self.lower_bound, self.upper_bound)