import numpy as np

class EnhancedChaoticLévyDE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = min(100, budget // 4)
        self.mutation_factor = 0.9  # Slightly increased mutation factor
        self.crossover_prob = 0.9  # Increased crossover probability for diversity
        self.local_search_prob = 0.3  # Enhanced local search probability
        self.tournament_size = 3
        self.population = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        self.fitness = np.full(self.population_size, float('inf'))
        self.evaluations = 0
        self.chaos_sequence = np.random.rand(self.population_size)

    def __call__(self, func):
        self.fitness = np.apply_along_axis(func, 1, self.population)
        self.evaluations += self.population_size

        while self.evaluations < self.budget:
            for i in range(self.population_size):
                if self.evaluations >= self.budget:
                    break

                indices = [idx for idx in range(self.population_size) if idx != i]
                chosen = np.random.choice(indices, self.tournament_size, replace=False)
                best_idx = min(chosen, key=lambda idx: self.fitness[idx])
                a, b, c = self.population[best_idx], self.population[np.random.choice(indices)], self.population[np.random.choice(indices)]
                
                mutation_factor_dynamic = self.mutation_factor * np.random.rand()
                mutant = np.clip(a + mutation_factor_dynamic * (b - c), self.lower_bound, self.upper_bound)
                
                cross_points = np.random.rand(self.dim) < self.crossover_prob
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                
                trial = np.where(cross_points, mutant, self.population[i])
                trial_fitness = func(trial)
                self.evaluations += 1
                
                if trial_fitness < self.fitness[i]:
                    self.population[i] = trial
                    self.fitness[i] = trial_fitness

                if np.random.rand() < self.local_search_prob * np.sqrt(1 - self.evaluations / self.budget):
                    self.chaotic_levy_local_search(i, func)
        
        best_idx = np.argmin(self.fitness)
        return self.population[best_idx], self.fitness[best_idx]

    def chaotic_levy_local_search(self, index, func):
        chaos_factor = 0.2  # Adjusted chaos factor for stability
        for _ in range(3):
            if self.evaluations >= self.budget:
                break

            self.chaos_sequence[index] = 4 * self.chaos_sequence[index] * (1 - self.chaos_sequence[index])
            levy_step = self.levy_flight(1.5, self.dim)  # Lévy flight for diverse exploration
            perturbation = chaos_factor * (self.upper_bound - self.lower_bound) * levy_step
            neighbor = np.clip(self.population[index] + perturbation, self.lower_bound, self.upper_bound)
            neighbor_fitness = func(neighbor)
            self.evaluations += 1
            
            if neighbor_fitness < self.fitness[index]:
                self.population[index] = neighbor
                self.fitness[index] = neighbor_fitness

    def levy_flight(self, beta, size):
        sigma_u = (np.math.gamma(1 + beta) * np.sin(np.pi * beta / 2) / 
                   (np.math.gamma((1 + beta) / 2) * beta * 2**((beta - 1) / 2)))**(1 / beta)
        u = np.random.normal(0, sigma_u, size)
        v = np.random.normal(0, 1, size)
        step = u / np.abs(v)**(1 / beta)
        return step