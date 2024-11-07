import numpy as np

class DynamicIMPHS(IMPHS):
    def __init__(self, budget, dim):
        super().__init__(budget, dim)
        self.min_population = int(0.5 * budget)
        self.max_population = int(1.5 * budget)

    def __call__(self, func):
        def adjust_population_size(population):
            evals = evaluate_population(population)
            avg_eval = np.mean(evals)
            std_eval = np.std(evals)
            if len(population) < self.max_population and std_eval < 0.1 * avg_eval:
                new_individual = np.random.uniform(self.lower_bound, self.upper_bound, self.dim)
                return np.vstack((population, new_individual))
            elif len(population) > self.min_population and std_eval > 0.3 * avg_eval:
                idx = np.argmax(evals)
                return np.delete(population, idx, axis=0)
            return population

        population = initialize_population()
        for _ in range(self.budget // 2):
            population = exploit_phase(explore_phase(population))
            population = adjust_population_size(population)
        best_idx = np.argmin(evaluate_population(population))
        best_solution = population[best_idx]
        return best_solution