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
        self.initial_temperature = 1000  # Initial temperature for simulated annealing
        self.cooling_rate = 0.95  # Cooling rate
        self.dynamic_adaptation_factor = 0.99  # Adaptation factor for dynamic parameter adjustment

    def __call__(self, func):
        np.random.seed(42)  # For reproducibility
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
                noise = np.random.uniform(-0.01, 0.01, self.dim)  # small noise for diversity
                mutant = np.clip(x0 + random_factor * self.F * (x1 - x2) + noise, self.lower_bound, self.upper_bound)
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

            # Simulated annealing step with adaptive temperature
            for i in range(self.population_size):
                perturbation = np.random.uniform(-0.1, 0.1, self.dim)
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

        return best