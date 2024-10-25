import numpy as np

class HybridDEPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = max(5, dim * 10)
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.scaling_factors = [0.5, 0.7, 0.9]
        self.crossover_rate = 0.9
        self.inertia_weight = 0.5
        self.cognitive_component = 1.5
        self.social_component = 1.5
        self.strategy_weights = np.ones(3)  # DE, PSO, Hybrid
        self.velocities = np.random.uniform(-1.0, 1.0, (self.population_size, self.dim))

    def __call__(self, func):
        population = np.random.uniform(
            self.lower_bound, self.upper_bound, (self.population_size, self.dim)
        )
        fitness = np.array([func(ind) for ind in population])
        eval_count = self.population_size
        personal_best = np.copy(population)
        personal_best_fitness = np.copy(fitness)
        global_best_idx = np.argmin(fitness)
        global_best = population[global_best_idx]

        while eval_count < self.budget:
            new_population = np.empty_like(population)
            new_fitness = np.empty(self.population_size)

            for i in range(self.population_size):
                strategy_idx = np.random.choice(
                    3, p=self.strategy_weights / self.strategy_weights.sum()
                )
                if strategy_idx == 0:
                    # Differential Evolution strategy
                    trial = self.de_rand_1_bin(population, i, global_best)
                elif strategy_idx == 1:
                    # Particle Swarm Optimization strategy
                    trial = self.pso_update(population[i], i, personal_best[i], global_best)
                else:
                    # Hybrid strategy
                    trial = self.hybrid_update(population[i], i, personal_best[i], global_best)
                trial = np.clip(trial, self.lower_bound, self.upper_bound)
                trial_fitness = func(trial)
                eval_count += 1

                if trial_fitness < fitness[i]:
                    new_population[i] = trial
                    new_fitness[i] = trial_fitness
                    personal_best[i] = trial
                    personal_best_fitness[i] = trial_fitness
                    self.strategy_weights[strategy_idx] += 1
                else:
                    new_population[i] = population[i]
                    new_fitness[i] = fitness[i]
            
            population = new_population
            fitness = new_fitness
            global_best_idx = np.argmin(fitness)
            global_best = population[global_best_idx]
        
        return global_best

    def de_rand_1_bin(self, population, idx, global_best):
        a, b, c = population[np.random.choice(range(self.population_size), 3, replace=False)]
        mutant = a + np.random.choice(self.scaling_factors) * (b - c)
        return self.binomial_crossover(population[idx], mutant)

    def pso_update(self, particle, idx, personal_best, global_best):
        velocity = (
            self.inertia_weight * self.velocities[idx]
            + self.cognitive_component * np.random.rand(self.dim) * (personal_best - particle)
            + self.social_component * np.random.rand(self.dim) * (global_best - particle)
        )
        self.velocities[idx] = velocity
        return particle + velocity

    def hybrid_update(self, particle, idx, personal_best, global_best):
        velocity = (
            self.inertia_weight * self.velocities[idx]
            + self.cognitive_component * np.random.rand(self.dim) * (personal_best - particle)
            + self.social_component * np.random.rand(self.dim) * (global_best - particle)
        )
        self.velocities[idx] = velocity
        mutant = personal_best + np.random.choice(self.scaling_factors) * (global_best - particle)
        de_candidate = self.binomial_crossover(particle, mutant)
        return (velocity + de_candidate) / 2

    def binomial_crossover(self, target, mutant):
        trial = np.empty_like(target)
        jrand = np.random.randint(self.dim)
        for j in range(self.dim):
            if np.random.rand() < self.crossover_rate or j == jrand:
                trial[j] = mutant[j]
            else:
                trial[j] = target[j]
        return trial