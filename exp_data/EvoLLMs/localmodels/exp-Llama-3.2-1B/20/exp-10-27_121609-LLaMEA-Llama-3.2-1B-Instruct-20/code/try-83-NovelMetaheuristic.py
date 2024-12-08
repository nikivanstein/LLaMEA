# One-Liner Description: Novel Metaheuristic Algorithm for Black Box Optimization
# Code: 
# 
# class NovelMetaheuristic:
#     def __init__(self, budget, dim):
#         self.budget = budget
#         self.dim = dim
#         self.population_size = 50
#         self.population = np.random.uniform(-5.0, 5.0, (self.population_size, self.dim))
#         self.fitnesses = np.zeros((self.population_size, self.dim))

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 50
        self.population = np.random.uniform(-5.0, 5.0, (self.population_size, self.dim))
        self.fitnesses = np.zeros((self.population_size, self.dim))
        self.replacement_rate = 0.2

    def __call__(self, func):
        def objective(x):
            return func(x)

        def bounds(x):
            return (x.min() - 5.0, x.max() + 5.0)

        def mutation(x):
            if np.random.rand() < self.replacement_rate:
                return np.random.uniform(bounds(x)[0], bounds(x)[1])
            return x

        def selection(x):
            return np.random.choice(self.population_size, 1, replace=False)

        for _ in range(self.budget):
            for i in range(self.population_size):
                x = self.population[i]
                fitness = objective(x)
                if fitness < self.fitnesses[i, x] + 1e-6:
                    self.fitnesses[i, x] = fitness
                    self.population[i] = mutation(x)

            # Select the fittest individual
            fittest_index = np.argmax(self.fitnesses)
            new_individual = self.population[selection(fittest_index)]

            # Update the population with the new individual
            self.population[fittest_index] = mutation(new_individual)
            self.population = np.array([mutation(x) for x in self.population])

        return self.fitnesses

# Example usage:
if __name__ == "__main__":
    algorithm = NovelMetaheuristic(1000, 10)
    func = lambda x: x**2
    best_solution = algorithm(func)
    print("Best solution:", best_solution)