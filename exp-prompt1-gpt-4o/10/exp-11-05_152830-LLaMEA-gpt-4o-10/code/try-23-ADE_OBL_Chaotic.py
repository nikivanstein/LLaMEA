import numpy as np

class ADE_OBL_Chaotic:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.bounds = (-5.0, 5.0)
        self.population_size = 20 + 5 * dim
        self.mutation_factor = 0.8
        self.crossover_prob = 0.9

    def _initialize_population(self):
        return np.random.uniform(self.bounds[0], self.bounds[1], (self.population_size, self.dim))

    def _evaluate_population(self, population, func):
        return np.array([func(ind) for ind in population])

    def _opposite_population(self, population):
        min_bound, max_bound = self.bounds
        return min_bound + max_bound - population

    def _chaotic_perturbation(self, individual):
        # Introducing a simple chaotic perturbation using a logistic map
        r = 3.99  # Chaotic parameter
        return individual + 0.5 * (r * individual * (1 - individual)) * np.random.uniform(-1, 1, size=self.dim)

    def __call__(self, func):
        population = self._initialize_population()
        fitness = self._evaluate_population(population, func)

        eval_count = self.population_size

        while eval_count < self.budget:
            # Opposition-based learning
            opposite_population = self._opposite_population(population)
            opposite_fitness = self._evaluate_population(opposite_population, func)

            combined_population = np.vstack((population, opposite_population))
            combined_fitness = np.hstack((fitness, opposite_fitness))

            # Select the best individuals from the combined population
            best_indices = np.argsort(combined_fitness)[:self.population_size]
            population = combined_population[best_indices]
            fitness = combined_fitness[best_indices]

            eval_count += self.population_size

            if eval_count >= self.budget:
                break

            new_population = np.zeros_like(population)
            
            for i in range(self.population_size):
                a, b, c = population[np.random.choice(self.population_size, 3, replace=False)]
                mutant = a + self.mutation_factor * (b - c)
                mutant = np.clip(mutant, self.bounds[0], self.bounds[1])

                cross_points = np.random.rand(self.dim) < self.crossover_prob
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True

                trial = np.where(cross_points, mutant, population[i])
                # Apply chaotic perturbation to the trial vector
                trial = self._chaotic_perturbation(trial)
                trial = np.clip(trial, self.bounds[0], self.bounds[1])
                
                trial_fitness = func(trial)
                eval_count += 1

                if trial_fitness < fitness[i]:
                    new_population[i] = trial
                    fitness[i] = trial_fitness
                else:
                    new_population[i] = population[i]

            population = new_population

        best_idx = np.argmin(fitness)
        return population[best_idx], fitness[best_idx]