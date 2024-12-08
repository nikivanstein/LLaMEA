# import numpy as np

class CustomHybridMetaheuristicAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10
        self.harmony_memory_size = 5
        self.iterations = 100
        self.de_mut_prob = 0.7
        self.de_cross_prob = 0.9
        self.firefly_gamma = 0.1
        self.pso_inertia_weight = 0.7
        self.pso_cognitive_weight = 1.5
        self.pso_social_weight = 1.5

    def __call__(self, func):
        def objective_function(x):
            return func(x)

        def pso_move(current, best):
            inertia_velocity = self.pso_inertia_weight * (current - best)
            cognitive_velocity = self.pso_cognitive_weight * np.random.rand(self.dim) * (best - current)
            social_velocity = self.pso_social_weight * np.random.rand(self.dim) * (best - current)
            new_position = current + inertia_velocity + cognitive_velocity + social_velocity
            return np.clip(new_position, -5.0, 5.0)

        def harmony_search():
            hm = np.random.uniform(-5.0, 5.0, (self.harmony_memory_size, self.dim))
            for _ in range(self.iterations):
                harmony_memory_fitness = np.array([objective_function(x) for x in hm])
                best_harmony = hm[np.argmin(harmony_memory_fitness)]
                new_harmony = np.clip(best_harmony + 0.01 * np.random.randn(self.dim), -5.0, 5.0)
                random_index = np.random.randint(self.harmony_memory_size)
                if objective_function(new_harmony) < harmony_memory_fitness[random_index]:
                    hm[random_index] = new_harmony
            return best_harmony

        def differential_evolution(population):
            mutated_population = []
            for i in range(self.population_size):
                indices = list(range(self.population_size))
                indices.remove(i)
                a, b, c = np.random.choice(indices, 3, replace=False)
                mutant = population[a] + self.de_mut_prob * (population[b] - population[c])
                crossover = np.random.rand(self.dim) < self.de_cross_prob
                trial = np.where(crossover, mutant, population[i])
                if objective_function(trial) < objective_function(population[i]):
                    mutated_population.append(trial)
                else:
                    mutated_population.append(population[i])
            return np.array(mutated_population)

        best_solution = np.random.uniform(-5.0, 5.0, self.dim)
        for _ in range(self.budget):
            population = [pso_move(best_solution, np.random.uniform(-5.0, 5.0, self.dim)) for _ in range(self.population_size)]
            population = differential_evolution(population)
            best_solution = population[np.argmin([objective_function(x) for x in population])]
            if np.random.rand() < 0.35:
                best_solution = pso_move(best_solution, harmony_search())
        return best_solution