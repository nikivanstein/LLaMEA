import numpy as np

class HybridDEFirefly:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = 20  # Initial population size
        self.mutation_factor = 0.5  # Differential evolution parameter
        self.crossover_rate = 0.7  # Crossover probability
        self.alpha = 0.5  # Firefly algorithm parameter
        self.gamma = 1.0  # Light absorption coefficient
        self.success_history = []

    def __call__(self, func):
        np.random.seed(42)  # For reproducibility
        budget_used = 0

        # Initialize population
        population = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        budget_used += self.population_size

        best_idx = np.argmin(fitness)
        best_solution = population[best_idx]
        best_fitness = fitness[best_idx]

        while budget_used < self.budget:
            new_population = np.copy(population)

            # Update crossover rate based on fitness variance
            fitness_variance = np.var(fitness)
            adaptive_crossover_rate = self.crossover_rate * (1 + np.tanh(fitness_variance))

            for i in range(self.population_size):
                # Differential Evolution mutation with enhanced dynamic mutation factor
                indices = list(range(self.population_size))
                indices.remove(i)
                a, b, c = np.random.choice(indices, 3, replace=False)
                dynamic_mutation_factor = self.mutation_factor * np.sqrt(1 - (budget_used / self.budget))
                if self.success_history:  # Adapt mutation factor based on success history
                    dynamic_mutation_factor *= (1 + np.mean(self.success_history))

                mutant = np.clip(population[a] + dynamic_mutation_factor * (population[b] - population[c]),
                                 self.lower_bound, self.upper_bound)

                # Crossover with adaptive crossover rate
                cross_points = np.random.rand(self.dim) < adaptive_crossover_rate
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                trial = np.where(cross_points, mutant, population[i])

                # Evaluate trial vector
                trial_fitness = func(trial)
                budget_used += 1

                # Selection
                if trial_fitness < fitness[i]:
                    new_population[i] = trial
                    fitness[i] = trial_fitness
                    self.success_history.append(0.1)  # Record success
                else:
                    self.success_history.append(-0.1)

            # Firefly algorithm component with adaptive parameters
            for i in range(self.population_size):
                for j in range(self.population_size):
                    if fitness[j] < fitness[i]:
                        r = np.linalg.norm(new_population[i] - new_population[j])
                        beta = np.exp(-self.gamma * r**2)
                        # Adaptive alpha for dynamic search balance
                        adaptive_alpha = self.alpha * (1 - (budget_used / self.budget))
                        new_population[i] += beta * (new_population[j] - new_population[i]) + adaptive_alpha * (np.random.rand(self.dim) - 0.5)

                        # Bound check
                        new_population[i] = np.clip(new_population[i], self.lower_bound, self.upper_bound)
                        new_fitness = func(new_population[i])
                        budget_used += 1
                        if new_fitness < fitness[i]:
                            fitness[i] = new_fitness

            # Update population
            population = new_population
            
            # Update best solution
            current_best_idx = np.argmin(fitness)
            if fitness[current_best_idx] < best_fitness:
                best_fitness = fitness[current_best_idx]
                best_solution = population[current_best_idx]

            # Adjust population size based on fitness diversity
            if np.std(fitness) < 1e-3:  # Population converged
                self.population_size = max(10, self.population_size - 2)

            if budget_used >= self.budget:
                break

        return best_solution, best_fitness