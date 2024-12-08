import numpy as np

class QuantumInspiredDifferentialEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.bounds = (-5.0, 5.0)
        self.population_size = max(12, 12 * dim // 3)  # Adjusted population size
        self.mutation_factor = 0.6
        self.crossover_rate = 0.85
        self.population = np.random.uniform(self.bounds[0], self.bounds[1], (self.population_size, dim))
        self.best_solution = None
        self.best_fitness = np.inf
        self.adaptive_sigma = 0.15
        self.mutation_strategy = [0.4, 0.9]  # Tweaked self-adaptive mutation factor

    def __call__(self, func):
        evaluations = 0
        improvement_counter = 0
        while evaluations < self.budget:
            new_population = np.empty_like(self.population)
            for i in range(self.population_size):
                if evaluations >= self.budget:
                    break
                indices = list(range(self.population_size))
                indices.remove(i)
                a, b, c = np.random.choice(indices, 3, replace=False)
                mutant = self.population[a] + np.random.choice(self.mutation_strategy) * (self.population[b] - self.population[c])
                mutant = np.clip(mutant, *self.bounds)
                
                trial = mutant if np.random.rand() < self.crossover_rate else self.population[i]
                
                # Quantum-inspired perturbation
                perturbation = np.random.normal(0, self.adaptive_sigma, self.dim) * np.random.choice([-1, 1], self.dim)
                trial_perturbed = trial + perturbation
                trial_perturbed = np.clip(trial_perturbed, *self.bounds)
                
                trial_fitness = func(trial_perturbed)
                evaluations += 1

                if trial_fitness < func(self.population[i]):
                    new_population[i] = trial_perturbed
                    improvement_counter = 0
                else:
                    new_population[i] = self.population[i]
                    improvement_counter += 1

                if trial_fitness < self.best_fitness:
                    self.best_fitness = trial_fitness
                    self.best_solution = trial_perturbed

            if improvement_counter > self.population_size // 3:
                self.adaptive_sigma = min(self.adaptive_sigma * 1.15, 1.0)
            else:
                self.adaptive_sigma = max(self.adaptive_sigma * 0.85, 0.01)

            self.population = new_population

        return self.best_solution, self.best_fitness