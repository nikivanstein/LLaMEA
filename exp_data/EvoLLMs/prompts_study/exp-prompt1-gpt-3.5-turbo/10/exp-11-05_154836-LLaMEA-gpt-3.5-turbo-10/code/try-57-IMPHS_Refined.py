import numpy as np

class IMPHS_Refined(IMPHS):
    def exploit_phase(self, population, num_iterations=5):
        F = 0.5
        CR = 0.9
        for _ in range(num_iterations):
            best_idx = np.argmin(self.evaluate_population(population))
            best_individual = population[best_idx]
            for i in range(len(population)):
                a, b, c = np.random.choice(np.delete(np.arange(len(population)), i, axis=0), 3, replace=False)
                trial_vector = population[a] + F * (population[b] - population[c])
                mask = np.random.rand(self.dim) < CR
                trial_vector = np.where(mask, trial_vector, population[i])
                trial_vector = np.clip(trial_vector, self.lower_bound, self.upper_bound)
                if func(trial_vector) < func(population[i]):
                    population[i] = trial_vector
            population[best_idx] = best_individual
        return population