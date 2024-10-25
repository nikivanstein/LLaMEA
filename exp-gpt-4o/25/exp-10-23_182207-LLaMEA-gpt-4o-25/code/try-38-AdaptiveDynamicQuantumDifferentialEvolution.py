import numpy as np

class AdaptiveDynamicQuantumDifferentialEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.bounds = (-5.0, 5.0)
        self.population_size = max(12, 12 * dim // 3)
        self.mutation_factor = 0.9
        self.crossover_rate = 0.85
        self.population = np.random.uniform(self.bounds[0], self.bounds[1], (self.population_size, dim))
        self.best_solution = None
        self.best_fitness = np.inf
        self.entanglement_factor = 0.05
        self.dynamic_mutation_factor = 0.07
        self.stochastic_tunneling_factor = 0.1

    def __call__(self, func):
        evaluations = 0
        stagnation_counter = 0
        while evaluations < self.budget:
            new_population = np.empty_like(self.population)
            fitness_values = np.apply_along_axis(func, 1, self.population)
            evaluations += self.population_size
            for i in range(self.population_size):
                if evaluations >= self.budget:
                    break
                indices = list(range(self.population_size))
                indices.remove(i)
                a, b, c = np.random.choice(indices, 3, replace=False)
                dynamic_mutation = self.mutation_factor + np.random.normal(0, self.dynamic_mutation_factor)
                mutant = self.population[a] + np.random.choice([0.5, 1.0]) * (self.population[b] - self.population[c])
                mutant = np.clip(mutant, *self.bounds)
                
                trial = np.where(np.random.rand(self.dim) < self.crossover_rate, mutant, self.population[i])
                
                entanglement = np.random.normal(0, self.entanglement_factor, self.dim)
                trial_entangled = trial + entanglement
                trial_entangled = np.clip(trial_entangled, *self.bounds)
                
                trial_fitness = func(trial_entangled)
                evaluations += 1

                if trial_fitness < fitness_values[i]:
                    new_population[i] = trial_entangled
                    stagnation_counter = 0
                else:
                    new_population[i] = self.population[i]
                    stagnation_counter += 1

                if trial_fitness < self.best_fitness:
                    self.best_fitness = trial_fitness
                    self.best_solution = trial_entangled

            if stagnation_counter > self.population_size // 2:
                self.entanglement_factor = min(self.entanglement_factor * 1.1, 0.2)
                self.stochastic_tunneling_factor = max(self.stochastic_tunneling_factor * 0.9, 0.01)
            else:
                self.entanglement_factor = max(self.entanglement_factor * 0.9, 0.01)
                self.stochastic_tunneling_factor = min(self.stochastic_tunneling_factor * 1.1, 0.2)

            self.population = new_population

        return self.best_solution, self.best_fitness