import numpy as np

class HybridSCA_SA_ACOBL_GM_DPS_ALF:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.initial_population_size = 50
        self.population_size = self.initial_population_size
        self.T = 1000
        self.alpha = 0.99
        self.population = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        self.best_individual = np.random.uniform(self.lower_bound, self.upper_bound, self.dim)
        self.best_fitness = np.inf
        self.chaotic_map = np.random.uniform(0, 1, self.dim)
        self.adaptive_prob = 0.2
        self.mutation_prob = 0.1
        self.levy_flight_scale = 1.0

    def opposition_based_learning(self, individual):
        opposite_individual = self.lower_bound + self.upper_bound - individual
        return opposite_individual

    def chaotic_opposition_based_learning(self, individual):
        chaotic_individual = self.lower_bound + (self.upper_bound - self.lower_bound) * self.chaotic_map
        self.chaotic_map = 4 * self.chaotic_map * (1 - self.chaotic_map)
        return chaotic_individual

    def levy_flight(self, individual):
        levy_flight = np.random.normal(0, self.levy_flight_scale, self.dim) / np.random.normal(0, self.levy_flight_scale, self.dim)
        levy_flight = individual + levy_flight
        return levy_flight

    def gaussian_mutation(self, individual):
        gaussian_mutation = individual + np.random.normal(0, 0.1, self.dim)
        return gaussian_mutation

    def __call__(self, func):
        evaluations = 0
        while evaluations < self.budget:
            for i in range(self.population_size):
                fitness = func(self.population[i])
                evaluations += 1
                if fitness < self.best_fitness:
                    self.best_fitness = fitness
                    self.best_individual = self.population[i]
                r1 = np.random.uniform(0, 1, self.dim)
                r2 = np.random.uniform(0, 1, self.dim)
                r3 = np.random.uniform(0, 1, self.dim)
                r4 = np.random.uniform(0, 1, self.dim)
                A = 2 * r1 - 1
                B = np.abs(r2)
                C = 2 * r3 - 1
                D = np.abs(r4)
                sine = np.abs(self.best_individual - self.population[i]) * np.sin(np.abs(r1) * np.pi)
                cosine = np.abs(self.best_individual - self.population[i]) * np.cos(np.abs(r1) * np.pi)
                if np.random.uniform(0, 1) < 0.5:
                    self.population[i] = self.best_individual - (A * sine + B * cosine) * C * D
                else:
                    self.population[i] = self.best_individual + (A * sine + B * cosine) * C * D
                levy_flight = self.levy_flight(self.population[i])
                levy_fitness = func(levy_flight)
                evaluations += 1
                if levy_fitness < fitness:
                    self.population[i] = levy_flight
                    self.levy_flight_scale *= 1.1  # Increase the Levy flight scale if the Levy flight improves the fitness
                else:
                    delta = levy_fitness - fitness
                    prob = np.exp(-delta / self.T)
                    self.T *= self.alpha
                    if np.random.uniform(0, 1) < prob:
                        self.population[i] = levy_flight
                        self.levy_flight_scale *= 0.9  # Decrease the Levy flight scale if the Levy flight does not improve the fitness
                self.levy_flight_scale = np.clip(self.levy_flight_scale, 0.1, 10.0)  # Clip the Levy flight scale to a reasonable range
                # Apply adaptive chaotic opposition-based learning with a probability that adapts to the convergence speed
                if np.random.uniform(0, 1) < self.adaptive_prob:
                    chaotic_individual = self.chaotic_opposition_based_learning(self.population[i])
                    chaotic_fitness = func(chaotic_individual)
                    evaluations += 1
                    if chaotic_fitness < fitness:
                        self.population[i] = chaotic_individual
                        self.adaptive_prob *= 1.1  # Increase the probability if the chaotic opposition-based learning improves the fitness
                    else:
                        self.adaptive_prob *= 0.9  # Decrease the probability if the chaotic opposition-based learning does not improve the fitness
                    self.adaptive_prob = np.clip(self.adaptive_prob, 0.1, 0.9)  # Clip the probability to a reasonable range
                # Apply Gaussian mutation with a fixed probability
                if np.random.uniform(0, 1) < self.mutation_prob:
                    gaussian_individual = self.gaussian_mutation(self.population[i])
                    gaussian_fitness = func(gaussian_individual)
                    evaluations += 1
                    if gaussian_fitness < fitness:
                        self.population[i] = gaussian_individual
            # Dynamically adjust the population size based on the convergence speed
            if evaluations > self.budget / 2 and self.population_size < self.initial_population_size * 2:
                self.population_size *= 2
                self.population = np.vstack((self.population, np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size // 2, self.dim))))
        return self.best_individual