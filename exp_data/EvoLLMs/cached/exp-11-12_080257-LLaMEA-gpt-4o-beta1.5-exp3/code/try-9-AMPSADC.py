import numpy as np

class AMPSADC:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.eval_count = 0
        self.num_populations = 3
        self.population_size = max(5, 2 * dim)
        self.initial_temp = 1.0
        self.final_temp = 0.01
        self.cooling_rate = 0.95

    def initialize_populations(self):
        return [np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
                for _ in range(self.num_populations)]

    def evaluate_population(self, population, func):
        fitness = np.array([func(ind) for ind in population])
        self.eval_count += len(population)
        return fitness

    def perturb(self, individual, temperature):
        noise = np.random.normal(scale=temperature, size=self.dim)
        candidate = np.clip(individual + noise, self.lower_bound, self.upper_bound)
        return candidate

    def acceptance_probability(self, candidate_fitness, current_fitness, temperature):
        if candidate_fitness < current_fitness:
            return 1.0
        else:
            return np.exp((current_fitness - candidate_fitness) / temperature)

    def adaptive_cooling(self, temperature, improvement):
        if improvement:
            return max(self.final_temp, temperature * self.cooling_rate)
        else:
            return min(1.0, temperature / self.cooling_rate)

    def __call__(self, func):
        populations = self.initialize_populations()
        temperature = self.initial_temp
        best_solutions = [None] * self.num_populations
        best_fitness = [float('inf')] * self.num_populations

        while self.eval_count < self.budget:
            for p in range(self.num_populations):
                population = populations[p]
                fitness = self.evaluate_population(population, func)
                for i in range(self.population_size):
                    if self.eval_count >= self.budget:
                        break
                    current_individual = population[i]
                    candidate = self.perturb(current_individual, temperature)
                    candidate_fitness = func(candidate)
                    self.eval_count += 1

                    ap = self.acceptance_probability(candidate_fitness, fitness[i], temperature)
                    if np.random.rand() < ap:
                        population[i] = candidate
                        fitness[i] = candidate_fitness

                best_idx = np.argmin(fitness)
                if fitness[best_idx] < best_fitness[p]:
                    best_solutions[p] = population[best_idx]
                    best_fitness[p] = fitness[best_idx]
                    temperature = self.adaptive_cooling(temperature, improvement=True)
                else:
                    temperature = self.adaptive_cooling(temperature, improvement=False)

            # Migrate best solutions across populations
            best_solution = min(best_solutions, key=lambda s: func(s))
            for p in range(self.num_populations):
                populations[p][np.random.randint(self.population_size)] = best_solution

        best_overall = min(best_solutions, key=lambda s: func(s))
        return best_overall