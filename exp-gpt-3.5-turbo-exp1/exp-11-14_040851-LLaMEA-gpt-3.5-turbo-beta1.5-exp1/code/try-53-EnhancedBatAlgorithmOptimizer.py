import numpy as np

class EnhancedBatAlgorithmOptimizer:
    def __init__(self, budget, dim, population_size=10, loudness=0.5, pulse_rate=0.5, alpha=0.9, gamma=0.9, differential_weight=0.5, crossover_rate=0.7, crossover_adjust_rate=0.1, mutation_scale=0.1):
        self.budget = budget
        self.dim = dim
        self.population_size = population_size
        self.loudness = loudness
        self.pulse_rate = pulse_rate
        self.alpha = alpha
        self.gamma = gamma
        self.differential_weight = differential_weight
        self.crossover_rate = crossover_rate
        self.crossover_adjust_rate = crossover_adjust_rate
        self.mutation_scale = mutation_scale

    def __call__(self, func):
        def init_population():
            return np.random.uniform(-5.0, 5.0, (self.population_size, self.dim))

        def update_frequency(f):
            return f * self.alpha

        def update_loudness(fitness_improved):
            if fitness_improved:
                return self.loudness * self.gamma
            else:
                return self.loudness / self.gamma

        def levy_flight():
            beta = 1.5
            sigma = (np.math.gamma(1 + beta) * np.sin(np.pi * beta / 2) / (
                    np.math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (1 / beta)
            u = np.random.randn(self.dim) * sigma
            v = np.random.randn(self.dim)
            step = u / abs(v) ** (1 / beta)
            return step

        def differential_evolution(population, fitness, func):
            new_population = np.copy(population)
            for i in range(self.population_size):
                idxs = np.arange(self.population_size)
                idxs = np.delete(idxs, i)
                a, b, c = np.random.choice(idxs, 3, replace=False)
                mutant = population[a] + self.differential_weight * (population[b] - population[c])
                crossover_mask = np.random.rand(self.dim) < self.crossover_rate
                trial = np.where(crossover_mask, mutant, population[i])
                if func(trial) < fitness[i]:
                    new_population[i] = trial
            return new_population

        population = init_population()
        fitness = np.array([func(x) for x in population])
        best_solution = population[np.argmin(fitness)]
        best_fitness = np.min(fitness)

        for _ in range(self.budget):
            new_population = differential_evolution(population, fitness, func)
            for i in range(self.population_size):
                if np.random.rand() > self.pulse_rate:
                    frequency = 0.0
                else:
                    frequency = update_frequency(0.0)
                    new_population[i] += levy_flight() * frequency

                if np.random.rand() < self.loudness and func(new_population[i]) < func(population[i]):
                    population[i] = new_population[i]
                    fitness[i] = func(population[i])
                    if fitness[i] < best_fitness:
                        best_solution = population[i]
                        best_fitness = fitness[i]
                        self.loudness = update_loudness(True)
                    else:
                        self.loudness = update_loudness(False)

                # Dynamically adjusting mutation scale based on population diversity
                if np.std(population) > 0.1:
                    self.mutation_scale *= 1.1
                else:
                    self.mutation_scale /= 1.1
                mutation = np.random.normal(0, self.mutation_scale, self.dim)
                new_population[i] += mutation

            if _ % int(0.2 * self.budget) == 0:
                mean_fitness = np.mean(fitness)
                std_fitness = np.std(fitness)
                if std_fitness < 0.1:
                    self.crossover_rate += self.crossover_adjust_rate
                elif std_fitness > 0.5:
                    self.crossover_rate -= self.crossover_adjust_rate
                self.crossover_rate = np.clip(self.crossover_rate, 0, 1)

        return best_solution