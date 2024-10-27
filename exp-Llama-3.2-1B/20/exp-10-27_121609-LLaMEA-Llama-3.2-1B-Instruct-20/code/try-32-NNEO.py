import numpy as np

class NNEO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 50
        self.population = np.random.uniform(-5.0, 5.0, (self.population_size, self.dim))
        self.fitnesses = np.zeros((self.population_size, self.dim))

    def __call__(self, func):
        def objective(x):
            return func(x)

        def bounds(x):
            return (x.min() - 5.0, x.max() + 5.0)

        def mutate(individual):
            if np.random.rand() < 0.2:  # probability of refinement
                idx = np.random.choice(self.dim, 1, replace=False)
                individual[idx] = np.random.uniform(-5.0, 5.0)
            return individual

        def evaluate_fitness(individual):
            fitness = objective(individual)
            if fitness < self.fitnesses[individual] + 1e-6:
                self.fitnesses[individual] = fitness
                return individual
            else:
                return individual

        for _ in range(self.budget):
            for i in range(self.population_size):
                individual = evaluate_fitness(self.population[i])
                if individual!= self.population[i]:  # avoid mutation of the same individual
                    self.population[i] = mutate(individual)

        return self.fitnesses

# BBOB test suite of 24 noiseless functions
def test_functions():
    # Define the test functions
    test_functions = {
        'f1': lambda x: np.sin(x),
        'f2': lambda x: np.cos(x),
        'f3': lambda x: x**2,
        'f4': lambda x: np.sin(x + 2*np.pi),
        'f5': lambda x: np.cos(x + 2*np.pi),
        'f6': lambda x: x**3,
        'f7': lambda x: np.sin(x + 4*np.pi),
        'f8': lambda x: np.cos(x + 4*np.pi),
        'f9': lambda x: x**4,
        'f10': lambda x: np.sin(x + 6*np.pi),
        'f11': lambda x: np.cos(x + 6*np.pi),
        'f12': lambda x: x**5,
        'f13': lambda x: np.sin(x + 8*np.pi),
        'f14': lambda x: np.cos(x + 8*np.pi),
        'f15': lambda x: x**6,
        'f16': lambda x: np.sin(x + 10*np.pi),
        'f17': lambda x: np.cos(x + 10*np.pi),
        'f18': lambda x: x**7,
        'f19': lambda x: np.sin(x + 12*np.pi),
        'f20': lambda x: np.cos(x + 12*np.pi),
        'f21': lambda x: x**8,
        'f22': lambda x: np.sin(x + 14*np.pi),
        'f23': lambda x: np.cos(x + 14*np.pi),
        'f24': lambda x: x**9
    }

    # Evaluate the test functions
    fitnesses = []
    for func_name, func in test_functions.items():
        fitness = func(test_functions[func_name](np.linspace(-10, 10, 100)))
        fitnesses.append(fitness)

    return fitnesses

# Run the BBOB test suite
fitnesses = test_functions()

# Initialize the NNEO algorithm
nneo = NNEO(100, 10)  # budget = 100, dimensionality = 10

# Optimize the test functions
fitnesses = nneo(__call__, test_functions)

# Print the fitness scores
print("Fitness scores:", fitnesses)

# Print the selected solution
selected_solution = nneo.population[fitnesses.index(max(fitnesses))]
print("Selected solution:", selected_solution)