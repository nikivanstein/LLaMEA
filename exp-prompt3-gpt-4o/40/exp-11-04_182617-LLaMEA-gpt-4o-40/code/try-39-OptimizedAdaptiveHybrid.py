import numpy as np

class OptimizedAdaptiveHybrid:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = 8 * dim  # Adjusted population size for exploration
        self.best_solution = None
        self.best_fitness = float('inf')
        self.eval_count = 0
        self.learning_rate = 0.05  # Adjusted learning rate for stability
        self.inertia_weight = 0.6  # Refined inertia for balance
        self.cognitive_const = 1.7  # Enhanced cognitive component
        self.social_const = 1.4  # Balanced social component
        self.mutation_factor = 0.8  # Differential Evolution mutation factor
        self.crossover_rate = 0.9  # Differential Evolution crossover rate
    
    def differential_evolution(self, population, fitness):
        new_population = np.copy(population)

        for i in range(self.population_size):
            indices = np.random.choice(self.population_size, 3, replace=False)
            x1, x2, x3 = population[indices]
            mutant_vector = np.clip(x1 + self.mutation_factor * (x2 - x3), self.lower_bound, self.upper_bound)
            trial_vector = np.copy(new_population[i])

            for j in range(self.dim):
                if np.random.rand() < self.crossover_rate:
                    trial_vector[j] = mutant_vector[j]
            
            trial_fitness = fitness(trial_vector)
            self.eval_count += 1

            if trial_fitness < fitness(new_population[i]):
                new_population[i] = trial_vector

            if trial_fitness < self.best_fitness:
                self.best_fitness = trial_fitness
                self.best_solution = trial_vector

        return new_population

    def adaptive_gradient_update(self, solution, fitness):
        gradient_direction = np.random.uniform(-1.0, 1.0, self.dim)
        gradient_step_size = self.learning_rate * (1 - self.eval_count / self.budget)
        adjusted_solution = np.clip(solution + gradient_step_size * gradient_direction, self.lower_bound, self.upper_bound)
        adjusted_fitness = fitness(adjusted_solution)
        self.eval_count += 1

        if adjusted_fitness < self.best_fitness:
            self.best_fitness = adjusted_fitness
            self.best_solution = adjusted_solution

    def __call__(self, func):
        population = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        fitness_values = np.array([func(ind) for ind in population])
        self.eval_count += self.population_size

        for i in range(self.population_size):
            if fitness_values[i] < self.best_fitness:
                self.best_fitness = fitness_values[i]
                self.best_solution = population[i]

        while self.eval_count < self.budget:
            population = self.differential_evolution(population, func)
            self.adaptive_gradient_update(self.best_solution, func)

        return self.best_solution