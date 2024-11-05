import numpy as np

class HybridDE_SA_Adaptive:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = 50
        self.F = 0.8  # Differential weight
        self.CR = 0.9  # Crossover probability
        self.initial_temperature = 1000
        self.cooling_rate = 0.93
        self.dynamic_adaptation_factor = 0.99

    def __call__(self, func):
        np.random.seed(42)
        population = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        evals = self.population_size

        best_idx = np.argmin(fitness)
        best = population[best_idx]
        best_fitness = fitness[best_idx]
        temperature = self.initial_temperature

        while evals < self.budget:
            new_population = np.empty_like(population)
            for i in range(self.population_size):
                indices = np.random.choice(self.population_size, 3, replace=False)
                x0, x1, x2 = population[indices]
                random_factor = np.random.uniform(0.5, 1.0)
                self.F = max(0.5, min(1.0, self.F + np.random.uniform(-0.1, 0.1)))  # Adjust F
                mutant = np.clip(x0 + random_factor * self.F * (x1 - x2), self.lower_bound, self.upper_bound)
                cross_points = np.random.rand(self.dim) < self.CR
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                trial = np.where(cross_points, mutant, population[i])
                trial_fitness = func(trial)
                evals += 1

                if trial_fitness < fitness[i]:
                    new_population[i] = trial
                    fitness[i] = trial_fitness
                else:
                    new_population[i] = population[i]

            population = new_population

            for i in range(self.population_size):
                perturbation = np.random.uniform(-0.05, 0.05, self.dim)
                candidate = np.clip(population[i] + perturbation, self.lower_bound, self.upper_bound)
                candidate_fitness = func(candidate)
                evals += 1
                
                if candidate_fitness < fitness[i] or np.random.rand() < np.exp((fitness[i] - candidate_fitness) / temperature):
                    population[i] = candidate
                    fitness[i] = candidate_fitness

            temperature *= self.cooling_rate
            self.cooling_rate *= self.dynamic_adaptation_factor

            current_best_idx = np.argmin(fitness)
            if fitness[current_best_idx] < best_fitness:
                best = population[current_best_idx]
                best_fitness = fitness[current_best_idx]

            # New elitism strategy
            if evals % (self.population_size * 2) == 0:
                elite_indices = fitness.argsort()[:5]  # Top 5 elites
                population[:5] = population[elite_indices]
                fitness[:5] = fitness[elite_indices]

        return best