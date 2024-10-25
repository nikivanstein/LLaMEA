import numpy as np

class AdaptiveEnsembleDifferentialEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.bounds = (-5.0, 5.0)
        self.population_size = max(10, 8 * dim // 3)  # Adjust population size
        self.mutation_factor = 0.9  # Changed mutation factor
        self.crossover_rate = 0.85  # Changed crossover rate
        self.population = np.random.uniform(self.bounds[0], self.bounds[1], (self.population_size, dim))
        self.best_solution = None
        self.best_fitness = np.inf
        self.adaptive_sigma = 0.1
        self.mutation_strategy = [0.5, 0.9, 1.5]  # Added additional mutation strategy
        self.strategy_probabilities = np.array([0.4, 0.4, 0.2])  # Probabilities for strategy selection

    def __call__(self, func):
        evaluations = 0
        stagnation_counter = 0
        while evaluations < self.budget:
            new_population = np.empty_like(self.population)
            for i in range(self.population_size):
                if evaluations >= self.budget:
                    break
                indices = list(range(self.population_size))
                indices.remove(i)
                a, b, c = np.random.choice(indices, 3, replace=False)
                strategy_choice = np.random.choice(len(self.mutation_strategy), p=self.strategy_probabilities)
                mutant = self.population[a] + self.mutation_strategy[strategy_choice] * (self.population[b] - self.population[c])
                mutant = np.clip(mutant, *self.bounds)
                
                trial = np.where(np.random.rand(self.dim) < self.crossover_rate, mutant, self.population[i])
                
                perturbation = np.random.normal(0, self.adaptive_sigma, self.dim)
                trial_perturbed = trial + perturbation
                trial_perturbed = np.clip(trial_perturbed, *self.bounds)
                
                trial_fitness = func(trial_perturbed)
                evaluations += 1

                if trial_fitness < func(self.population[i]):
                    new_population[i] = trial_perturbed
                    stagnation_counter = 0
                else:
                    new_population[i] = self.population[i]
                    stagnation_counter += 1
                
                if trial_fitness < self.best_fitness:
                    self.best_fitness = trial_fitness
                    self.best_solution = trial_perturbed

            if stagnation_counter > self.population_size // 3:  # Adjusted stagnation condition
                self.adaptive_sigma = min(self.adaptive_sigma * 1.3, 1.0)  # Adjusted sigma increase rate
            else:
                self.adaptive_sigma = max(self.adaptive_sigma * 0.7, 0.01)  # Adjusted sigma decrease rate

            # Update strategy probabilities based on performance
            successful_strategies = [i for i in range(len(self.mutation_strategy)) 
                                     if func(new_population[i]) < func(self.population[i])]
            if successful_strategies:
                for idx in successful_strategies:
                    self.strategy_probabilities[idx] += 0.05
                self.strategy_probabilities /= self.strategy_probabilities.sum()

            self.population = new_population

        return self.best_solution, self.best_fitness