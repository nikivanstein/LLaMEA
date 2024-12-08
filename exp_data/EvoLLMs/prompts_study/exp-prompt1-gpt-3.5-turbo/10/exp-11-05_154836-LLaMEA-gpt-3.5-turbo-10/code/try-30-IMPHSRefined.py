import numpy as np

class IMPHSRefined(IMPHS):
    def exploit_phase(self, population, num_iterations=5):
        for _ in range(num_iterations):
            best_idx = np.argmin(self.evaluate_population(population))
            best_individual = population[best_idx]
            for i in range(len(population)):
                idxs = np.random.choice(len(population), 3, replace=False)
                a, b, c = population[idxs]
                mutant = np.clip(a + 0.8 * (b - c), self.lower_bound, self.upper_bound)
                crossover_prob = np.random.rand(self.dim) < 0.9
                trial = np.where(crossover_prob, mutant, population[i])
                if func(trial) < func(population[i]):
                    population[i] = trial
            population[best_idx] = best_individual
        return population