import numpy as np

class EnhancedEnsembleDynamicDE(MultiDynamicDE):
    def __call__(self, func):
        populations = [np.random.uniform(-5.0, 5.0, (self.budget // self.num_populations, self.dim)) for _ in range(self.num_populations)]
        best_solutions = np.zeros((self.num_populations, self.dim))
        best_fitness = np.ones(self.num_populations) * np.inf
        
        for i in range(self.num_populations):
            fitness = np.array([func(x) for x in populations[i]])
            best_idx = np.argmin(fitness)
            best_solutions[i] = populations[i][best_idx]
            best_fitness[i] = fitness[best_idx]
        
        for _ in range(self.budget - self.num_populations):
            for i in range(self.num_populations):
                best_idx = np.argmin(best_fitness)
                mutants = [populations[i][np.random.choice(len(populations[i]), 3, replace=False) for _ in range(3)]

                # Ensemble mutation combining multiple strategies
                mutated_vectors = [best_solutions[i] + self.mutation_factors[i] * (mutants[j][0] - mutants[j][1]) * self.adaptive_probs[i] for j in range(3)]
                mutated_vectors = [np.clip(vec, -5.0, 5.0) for vec in mutated_vectors]
                trial_vectors = [np.where(np.random.rand(self.dim) < self.mutation_factors[i], mutated_vectors[j], best_solutions[i]) for j in range(3)]
                
                trial_fitness = [func(vec) for vec in trial_vectors]
                best_trial_idx = np.argmin(trial_fitness)
                
                if trial_fitness[best_trial_idx] < best_fitness[i]:
                    populations[i][best_idx] = trial_vectors[best_trial_idx]
                    best_solutions[i] = trial_vectors[best_trial_idx]
                    best_fitness[i] = trial_fitness[best_trial_idx]

                if np.random.rand() < 0.1:  # Update mutation factor and adaptive probabilities
                    self.mutation_factors[i] = np.clip(self.mutation_factors[i] * np.random.uniform(0.8, 1.2), self.mutation_factor_range[0], self.mutation_factor_range[1])
                    self.adaptive_probs[i] = np.clip(self.adaptive_probs[i] * np.random.uniform(0.8, 1.2), 0, 1)
        
        overall_best_idx = np.argmin(best_fitness)
        return best_solutions[overall_best_idx]