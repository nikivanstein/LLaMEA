import numpy as np

class HybridAPSO_QIDE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = 12 * dim  # Slightly smaller population for focused search
        self.population = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, dim))
        self.fitness = np.full(self.population_size, np.inf)
        self.scale_factor_low = 0.3  # Modified for more effective exploitation
        self.scale_factor_high = 0.7  # Narrowed range for better control
        self.crossover_rate = 0.85  # Slightly reduced to balance recombination
        self.evaluations = 0
        self.local_search_rate = 0.25  # Increased for better local refinement
        self.velocity = np.random.uniform(-1, 1, (self.population_size, dim))  # Added velocity for PSO dynamics
        self.inertia_weight = 0.5  # Inertia weight for PSO
        self.cognitive_coeff = 1.5  # Cognitive coefficient for PSO
        self.social_coeff = 1.5  # Social coefficient for PSO

    def __call__(self, func):
        self.evaluate_population(func)
        while self.evaluations < self.budget:
            for i in range(self.population_size):
                if self.evaluations >= self.budget:
                    break
                trial_vector = self.mutate(i)
                trial_vector = self.crossover(trial_vector, self.population[i])
                if np.random.rand() < self.local_search_rate:
                    trial_vector = self.local_search(trial_vector, func)
                    
                # PSO Update
                personal_best = self.population[i].copy()
                global_best = self.population[np.argmin(self.fitness)]
                self.velocity[i] = (self.inertia_weight * self.velocity[i] +
                                    self.cognitive_coeff * np.random.rand() * (personal_best - self.population[i]) +
                                    self.social_coeff * np.random.rand() * (global_best - self.population[i]))
                self.population[i] = np.clip(self.population[i] + self.velocity[i], self.lower_bound, self.upper_bound)

                trial_fitness = func(trial_vector)
                self.evaluations += 1
                if trial_fitness < self.fitness[i]:
                    self.fitness[i] = trial_fitness
                    self.population[i] = trial_vector
        
        best_idx = np.argmin(self.fitness)
        return self.population[best_idx]

    def evaluate_population(self, func):
        for i in range(self.population_size):
            if self.evaluations < self.budget:
                self.fitness[i] = func(self.population[i])
                self.evaluations += 1

    def mutate(self, idx):
        indices = [i for i in range(self.population_size) if i != idx]
        a, b, c = np.random.choice(indices, 3, replace=False)
        adaptive_scale_factor = np.random.uniform(self.scale_factor_low, self.scale_factor_high)
        mutant_vector = self.population[a] + adaptive_scale_factor * (self.population[b] - self.population[c])
        return np.clip(mutant_vector, self.lower_bound, self.upper_bound)

    def crossover(self, mutant_vector, target_vector):
        crossover = np.random.rand(self.dim) < self.crossover_rate
        if not np.any(crossover):
            crossover[np.random.randint(0, self.dim)] = True
        trial_vector = np.where(crossover, mutant_vector, target_vector)
        return trial_vector

    def local_search(self, vector, func):
        step_size = np.random.normal(0, 0.05, self.dim)  # Finer step size for precise local search
        local_vector = vector + step_size
        local_vector = np.clip(local_vector, self.lower_bound, self.upper_bound)
        if np.random.rand() < 0.5 and func(local_vector) < func(vector):
            vector = local_vector
        return vector