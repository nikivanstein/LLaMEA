import numpy as np

class HybridDEPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.bounds = (-5.0, 5.0)
        self.population_size = max(10, 10 * dim // 3)
        self.mutation_factor = 0.9  # Slightly higher to encourage diversity
        self.crossover_rate = 0.85  # Adjusted for balance
        self.population = np.random.uniform(self.bounds[0], self.bounds[1], (self.population_size, dim))
        self.best_solution = None
        self.best_fitness = np.inf
        self.velocities = np.random.uniform(-1, 1, (self.population_size, dim))  # Initial velocities for PSO
        self.individual_best_positions = np.copy(self.population)
        self.individual_best_fitness = np.inf * np.ones(self.population_size)
        self.global_best_position = None
        self.inertia_weight = 0.7
        self.cognitive_coefficient = 1.5
        self.social_coefficient = 1.5
        self.learning_rate = 0.05
        self.min_learning_rate = 0.01
        self.max_learning_rate = 0.3  # Increased for potentially faster adaptation

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
                dynamic_mutation = self.mutation_factor + np.random.normal(0, 0.02)
                mutant = self.population[a] + dynamic_mutation * (self.population[b] - self.population[c])
                mutant = np.clip(mutant, *self.bounds)
                
                trial = np.where(np.random.rand(self.dim) < self.crossover_rate, mutant, self.population[i])
                
                perturbation = np.random.normal(0, 0.1, self.dim)
                trial_perturbed = trial + self.learning_rate * perturbation
                trial_perturbed = np.clip(trial_perturbed, *self.bounds)
                
                trial_fitness = func(trial_perturbed)
                evaluations += 1

                if trial_fitness < fitness_values[i]:
                    new_population[i] = trial_perturbed
                    stagnation_counter = 0
                else:
                    new_population[i] = self.population[i]
                    stagnation_counter += 1

                if trial_fitness < self.individual_best_fitness[i]:
                    self.individual_best_fitness[i] = trial_fitness
                    self.individual_best_positions[i] = trial_perturbed

                if trial_fitness < self.best_fitness:
                    self.best_fitness = trial_fitness
                    self.best_solution = trial_perturbed
                    self.global_best_position = trial_perturbed

            for i in range(self.population_size):
                r1, r2 = np.random.rand(2)
                self.velocities[i] = (self.inertia_weight * self.velocities[i] +
                                      self.cognitive_coefficient * r1 * (self.individual_best_positions[i] - self.population[i]) +
                                      self.social_coefficient * r2 * (self.global_best_position - self.population[i]))
                new_population[i] = self.population[i] + self.velocities[i]
                new_population[i] = np.clip(new_population[i], *self.bounds)

            if stagnation_counter > self.population_size // 2:
                self.learning_rate = min(self.learning_rate * 1.2, self.max_learning_rate)
            else:
                self.learning_rate = max(self.learning_rate * 0.9, self.min_learning_rate)

            self.population = new_population

        return self.best_solution, self.best_fitness