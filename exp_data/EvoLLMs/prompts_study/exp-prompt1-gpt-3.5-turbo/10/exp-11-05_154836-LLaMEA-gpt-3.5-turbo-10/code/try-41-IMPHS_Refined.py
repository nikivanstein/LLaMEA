import numpy as np

class IMPHS_Refined(IMPHS):
    def exploit_phase(self, population, num_iterations=5):
        for _ in range(num_iterations):
            best_idx = np.argmin(self.evaluate_population(population))
            best_individual = population[best_idx]
            for i in range(len(population)):
                candidates = [ind for idx, ind in enumerate(population) if idx != i]
                a, b, c = np.random.choice(candidates, 3, replace=False)
                mutant = np.clip(a + 0.5 * (b - c), self.lower_bound, self.upper_bound)
                crossover_mask = np.random.rand(self.dim) < 0.9
                new_individual = np.where(crossover_mask, mutant, population[i])
                if func(new_individual) < func(population[i]):
                    population[i] = new_individual
        return population