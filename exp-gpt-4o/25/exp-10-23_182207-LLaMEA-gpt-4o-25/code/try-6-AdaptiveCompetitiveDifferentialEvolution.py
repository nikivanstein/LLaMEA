import numpy as np

class AdaptiveCompetitiveDifferentialEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.bounds = (-5.0, 5.0)
        self.population_size = max(20, 10 * dim // 2)  # More extensive population for diversity
        self.mutation_factor = 0.5 + np.random.rand() * 0.3  # Randomized mutation factor for diversity
        self.crossover_rate = 0.9
        self.population = np.random.uniform(self.bounds[0], self.bounds[1], (self.population_size, dim))
        self.best_solution = None
        self.best_fitness = np.inf
        self.adaptive_sigma = 0.1  # Start with a fixed sigma for Gaussian perturbation
        self.competition_frequency = 5  # Frequency of conducting competitive learning

    def __call__(self, func):
        evaluations = 0
        stagnation_counter = 0  # To track stagnation and adjust perturbation
        while evaluations < self.budget:
            new_population = np.empty_like(self.population)
            for i in range(self.population_size):
                if evaluations >= self.budget:
                    break
                indices = list(range(self.population_size))
                indices.remove(i)
                a, b, c = np.random.choice(indices, 3, replace=False)
                mutant = self.population[a] + self.mutation_factor * (self.population[b] - self.population[c])
                mutant = np.clip(mutant, *self.bounds)
                
                trial = np.where(np.random.rand(self.dim) < self.crossover_rate, mutant, self.population[i])
                
                # Adaptive Gaussian perturbation
                perturbation = np.random.normal(0, self.adaptive_sigma, self.dim)
                trial_perturbed = trial + perturbation
                trial_perturbed = np.clip(trial_perturbed, *self.bounds)
                
                trial_fitness = func(trial_perturbed)
                evaluations += 1

                if trial_fitness < func(self.population[i]):
                    new_population[i] = trial_perturbed
                    stagnation_counter = 0  # Reset stagnation counter
                else:
                    new_population[i] = self.population[i]
                    stagnation_counter += 1

                if trial_fitness < self.best_fitness:
                    self.best_fitness = trial_fitness
                    self.best_solution = trial_perturbed

            # Adjust perturbation based on stagnation
            if stagnation_counter > self.population_size // 2:
                self.adaptive_sigma = min(self.adaptive_sigma * 1.1, 1.0)  # Increase sigma
            else:
                self.adaptive_sigma = max(self.adaptive_sigma * 0.9, 0.01)  # Decrease sigma

            # Apply competitive learning to adjust mutation factor
            if evaluations % self.competition_frequency == 0:
                fitness_scores = np.array([func(ind) for ind in self.population])
                best_indices = fitness_scores.argsort()[:self.population_size // 3]
                # Increase mutation factor for less fit individuals
                for idx in range(self.population_size):
                    if idx not in best_indices:
                        self.mutation_factor = min(0.9, self.mutation_factor + 0.05)
                    else:
                        self.mutation_factor = max(0.4, self.mutation_factor - 0.05)

            self.population = new_population

        return self.best_solution, self.best_fitness