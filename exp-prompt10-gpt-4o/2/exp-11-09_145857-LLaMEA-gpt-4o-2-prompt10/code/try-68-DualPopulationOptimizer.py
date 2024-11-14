import numpy as np

class DualPopulationOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.initial_population_size = 20
        self.population_size = self.initial_population_size
        self.crossover_rate = 0.7
        self.F = 0.8
        self.temperature = 1000
        self.cooling_rate = 0.99
        self.mutation_scale = 0.5
        self.elitism_rate = 0.1
        self.selection_pressure = 1.5
    
    def differential_evolution(self, population, scores):
        new_population = np.copy(population)
        mean_score = np.mean(scores)
        diversity_factor = np.std(scores) / mean_score
        dynamic_crossover_rate = self.crossover_rate + (0.25 * diversity_factor)
        
        num_elites = int(self.elitism_rate * self.population_size)
        elite_indices = np.argsort(scores)[:max(1, num_elites)]
        elites = population[elite_indices]
        
        for i in range(num_elites, self.population_size):
            candidates = list(range(self.population_size))
            candidates.remove(i)
            a, b, c = np.random.choice(candidates, 3, replace=False)
            x_t = population[a] + self.F * (population[b] - population[c])
            x_t = np.clip(x_t, self.lower_bound, self.upper_bound)
            crossover = np.random.rand(self.dim) < dynamic_crossover_rate
            if not np.any(crossover):
                crossover[np.random.randint(0, self.dim)] = True
            new_individual = np.where(crossover, x_t, population[i])
            new_score = self.func(new_individual)
            if new_score < scores[i]:
                new_population[i] = new_individual
                scores[i] = new_score
                self.mutation_scale = min(self.mutation_scale + 0.01, 1.0)
            else:
                self.mutation_scale = max(self.mutation_scale - 0.005 * self.selection_pressure, 0.1)
        
        new_population[:num_elites] = elites
        return new_population, scores

    def simulated_annealing(self, best_individual, best_score):
        current_individual = np.copy(best_individual)
        current_score = best_score
        for _ in range(self.population_size):
            candidate = current_individual + np.random.normal(0, self.mutation_scale, self.dim)
            candidate = np.clip(candidate, self.lower_bound, self.upper_bound)
            candidate_score = self.func(candidate)
            acceptance_prob = np.exp((current_score - candidate_score) / self.temperature)
            if candidate_score < current_score or np.random.rand() < acceptance_prob:
                current_individual = candidate
                current_score = candidate_score
            self.temperature *= self.cooling_rate
        return current_individual, current_score
    
    def resize_population(self, evals):
        if evals > self.budget / 2:
            self.population_size = max(10, self.initial_population_size // 2)

    def __call__(self, func):
        self.func = func
        population = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        scores = np.array([self.func(ind) for ind in population])
        best_idx = np.argmin(scores)
        best_individual = population[best_idx]
        best_score = scores[best_idx]
        evals = self.population_size
        
        while evals < self.budget:
            self.resize_population(evals)
            population, scores = self.differential_evolution(population, scores)
            evals += self.population_size
            if evals < self.budget:
                best_individual, best_score = self.simulated_annealing(best_individual, best_score)
                evals += self.population_size
        
        return best_individual