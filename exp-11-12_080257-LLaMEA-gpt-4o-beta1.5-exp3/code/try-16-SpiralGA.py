import numpy as np

class SpiralGA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = max(5, 3 * dim)
        self.eval_count = 0
        self.alpha = 0.1  # Spiral expansion factor
        self.beta = 0.5   # Spiral contraction factor

    def initialize_population(self):
        return np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))

    def evaluate_population(self, population, func):
        fitness = np.array([func(ind) for ind in population])
        self.eval_count += len(population)
        return fitness

    def spiral_mutation(self, center, point):
        direction = point - center
        angle = np.random.uniform(0, 2 * np.pi)
        radius = np.linalg.norm(direction)
        spiral_point = center + radius * np.array([np.cos(angle), np.sin(angle)])
        spiral_point = np.clip(spiral_point, self.lower_bound, self.upper_bound)
        return spiral_point

    def __call__(self, func):
        population = self.initialize_population()
        fitness = self.evaluate_population(population, func)

        while self.eval_count < self.budget:
            best_idx = np.argmin(fitness)
            best = population[best_idx]

            new_population = np.empty_like(population)
            for i in range(self.population_size):
                if self.eval_count >= self.budget:
                    break
                spiral_point = self.spiral_mutation(best, population[i])
                trial_fitness = func(spiral_point)
                self.eval_count += 1
                if trial_fitness < fitness[i]:
                    new_population[i] = spiral_point
                    fitness[i] = trial_fitness
                else:
                    new_population[i] = population[i]

            population = new_population

            # Expand and contract spirals
            self.alpha *= 1.1
            self.beta *= 0.9

        return population[np.argmin(fitness)]