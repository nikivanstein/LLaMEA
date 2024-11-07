import numpy as np

class DynamicIMPHS(IMPHS):
    def __call__(self, func):
        def dynamic_population_update(population, func_evaluations):
            if func_evaluations % 100 == 0 and len(population) > 2:
                keep_indices = np.argsort(evaluate_population(population))[:len(population)//2]
                return population[keep_indices]
            return population

        population = initialize_population()
        func_evaluations = 0
        for _ in range(self.budget // 2):
            population = dynamic_population_update(population, func_evaluations)
            population = exploit_phase(explore_phase(population))
            func_evaluations += len(population)
        best_idx = np.argmin(evaluate_population(population))
        best_solution = population[best_idx]
        return best_solution