import numpy as np

class AdaptiveSwarmDifferentialEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.bounds = (-5.0, 5.0)
        self.population_size = max(15, 10 * dim // 2)  # Adjusted population size for diversity
        self.initial_mutation_factor = 0.8
        self.initial_crossover_rate = 0.9
        self.population = np.random.uniform(self.bounds[0], self.bounds[1], (self.population_size, dim))
        self.best_solution = None
        self.best_fitness = np.inf
        self.adaptive_sigma = 0.1
        self.mutation_strategy = [0.5, 1.5]  # Expanded range for self-adaptive mutation factor
        self.velocities = np.random.uniform(-1, 1, (self.population_size, dim))
        self.personal_best_positions = self.population.copy()
        self.personal_best_fitness = np.full(self.population_size, np.inf)

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
                mutant = self.population[a] + np.random.choice(self.mutation_strategy) * (self.population[b] - self.population[c])
                mutant = np.clip(mutant, *self.bounds)
                
                crossover_rate = self.initial_crossover_rate * (1 - evaluations / self.budget) 
                trial = np.where(np.random.rand(self.dim) < crossover_rate, mutant, self.population[i])
                
                perturbation = np.random.normal(0, self.adaptive_sigma, self.dim)
                trial_perturbed = trial + perturbation
                trial_perturbed = np.clip(trial_perturbed, *self.bounds)
                
                trial_fitness = func(trial_perturbed)
                evaluations += 1

                if trial_fitness < func(self.population[i]):
                    new_population[i] = trial_perturbed
                    stagnation_counter = 0
                    if trial_fitness < self.personal_best_fitness[i]:
                        self.personal_best_fitness[i] = trial_fitness
                        self.personal_best_positions[i] = trial_perturbed
                else:
                    new_population[i] = self.population[i]
                    stagnation_counter += 1

                if trial_fitness < self.best_fitness:
                    self.best_fitness = trial_fitness
                    self.best_solution = trial_perturbed

            if stagnation_counter > self.population_size // 2:
                self.adaptive_sigma = min(self.adaptive_sigma * 1.2, 1.0)
            else:
                self.adaptive_sigma = max(self.adaptive_sigma * 0.8, 0.01)

            # Update velocities and move particles
            inertia = 0.5 + np.random.random() / 2
            cognitive = 1.5
            social = 1.5
            for i in range(self.population_size):
                self.velocities[i] = (
                    inertia * self.velocities[i] +
                    cognitive * np.random.random(self.dim) * (self.personal_best_positions[i] - self.population[i]) +
                    social * np.random.random(self.dim) * (self.best_solution - self.population[i])
                )
                new_population[i] = self.population[i] + self.velocities[i]
                new_population[i] = np.clip(new_population[i], *self.bounds)

            self.population = new_population

        return self.best_solution, self.best_fitness