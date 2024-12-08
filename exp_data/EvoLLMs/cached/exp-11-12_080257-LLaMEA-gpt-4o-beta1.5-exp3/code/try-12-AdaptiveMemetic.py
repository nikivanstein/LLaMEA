import numpy as np

class AdaptiveMemetic:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.init_population_size = max(5, 3 * dim)
        self.population_size = self.init_population_size
        self.eval_count = 0
        self.mutation_factor = 0.8
        self.crossover_rate = 0.7
        self.convergence_threshold = 0.01

    def initialize_population(self):
        return np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))

    def evaluate_population(self, population, func):
        fitness = np.array([func(ind) for ind in population])
        self.eval_count += len(population)
        return fitness

    def adapt_population_size(self, previous_best, current_best):
        if np.abs(previous_best - current_best) < self.convergence_threshold:
            self.population_size = min(self.population_size + 1, self.budget - self.eval_count)
        else:
            self.population_size = max(self.init_population_size, self.population_size - 1)

    def mutate(self, population, best):
        indices = np.random.choice(self.population_size, 3, replace=False)
        x1, x2, x3 = population[indices]
        mutant = x1 + self.mutation_factor * (x2 - x3)
        mutant = np.clip(mutant, self.lower_bound, self.upper_bound)
        return mutant

    def crossover(self, target, mutant):
        cross_points = np.random.rand(self.dim) < self.crossover_rate
        if not np.any(cross_points):
            cross_points[np.random.randint(0, self.dim)] = True
        trial = np.where(cross_points, mutant, target)
        return trial

    def enhanced_local_search(self, individual, func, intensity):
        epsilon = 0.1 * intensity
        neighbors = np.clip(individual + epsilon * np.random.uniform(-1, 1, (5, self.dim)), self.lower_bound, self.upper_bound)
        neighbor_fitness = self.evaluate_population(neighbors, func)
        best_idx = np.argmin(neighbor_fitness)
        return neighbors[best_idx], neighbor_fitness[best_idx]

    def __call__(self, func):
        population = self.initialize_population()
        fitness = self.evaluate_population(population, func)
        previous_best = np.inf
        
        while self.eval_count < self.budget:
            best_idx = np.argmin(fitness)
            best = population[best_idx]
            current_best_fitness = fitness[best_idx]
            
            self.adapt_population_size(previous_best, current_best_fitness)
            previous_best = current_best_fitness
            
            new_population = population.copy()
            new_fitness = fitness.copy()

            for i in range(self.population_size):
                if self.eval_count >= self.budget:
                    break
                mutant = self.mutate(population, best)
                trial = self.crossover(population[i], mutant)
                trial_fitness = func(trial)
                self.eval_count += 1
                if trial_fitness < fitness[i]:
                    new_population[i] = trial
                    new_fitness[i] = trial_fitness
            
            # Apply enhanced local search based on convergence
            intensity = 1 + (1 - current_best_fitness / fitness.max())
            for i in range(self.population_size):
                if self.eval_count >= self.budget:
                    break
                improved, improved_fitness = self.enhanced_local_search(new_population[i], func, intensity)
                if improved_fitness < new_fitness[i]:
                    new_population[i] = improved
                    new_fitness[i] = improved_fitness

            population, fitness = new_population, new_fitness

        return population[np.argmin(fitness)]