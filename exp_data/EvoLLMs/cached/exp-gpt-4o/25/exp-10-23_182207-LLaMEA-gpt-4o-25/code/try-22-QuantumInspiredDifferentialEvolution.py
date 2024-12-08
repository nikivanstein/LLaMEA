import numpy as np

class QuantumInspiredDifferentialEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.bounds = (-5.0, 5.0)
        self.population_size = max(10, 10 * dim // 3)
        self.mutation_factor = 0.8
        self.crossover_rate = 0.9
        self.population = np.random.uniform(self.bounds[0], self.bounds[1], (self.population_size, dim))
        self.best_solution = None
        self.best_fitness = np.inf
        self.adaptive_sigma = 0.1
        self.mutation_strategy = [0.5, 1.0]
        self.rotation_angle = 0.05

    def __call__(self, func):
        evaluations = 0
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
                
                quantum_prob = np.random.rand(self.dim)
                quantum_rotation = np.sin(quantum_prob * self.rotation_angle)
                trial = np.where(quantum_prob < self.crossover_rate, mutant * quantum_rotation + self.population[i] * (1 - quantum_rotation), self.population[i])
                
                perturbation = np.random.normal(0, self.adaptive_sigma, self.dim)
                trial_perturbed = trial + perturbation
                trial_perturbed = np.clip(trial_perturbed, *self.bounds)
                
                trial_fitness = func(trial_perturbed)
                evaluations += 1

                if trial_fitness < func(self.population[i]):
                    new_population[i] = trial_perturbed
                else:
                    new_population[i] = self.population[i]

                if trial_fitness < self.best_fitness:
                    self.best_fitness = trial_fitness
                    self.best_solution = trial_perturbed

            self.population = new_population

            mean_position = np.mean(self.population, axis=0)
            dist_to_mean = np.linalg.norm(self.population - mean_position, axis=1)
            self.adaptive_sigma = np.clip(np.mean(dist_to_mean) / np.sqrt(self.dim), 0.01, 0.5)

        return self.best_solution, self.best_fitness