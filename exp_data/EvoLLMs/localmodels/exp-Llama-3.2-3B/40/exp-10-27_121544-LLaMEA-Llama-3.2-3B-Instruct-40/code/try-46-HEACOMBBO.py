import numpy as np
from scipy.optimize import differential_evolution

class HEACOMBBO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.bounds = [(-5.0, 5.0)] * dim
        self.x0 = np.random.uniform(self.bounds[0][0], self.bounds[0][1], self.dim)

    def __call__(self, func):
        if self.budget == 0:
            return np.nan, np.nan

        if func.__name__ == 'f1':
            return differential_evolution(func, self.bounds)
        elif func.__name__ == 'f2':
            return differential_evolution(func, self.bounds, x0=self.x0)
        elif func.__name__ == 'f3':
            return differential_evolution(func, self.bounds, x0=self.x0, tol=1e-5)
        elif func.__name__ == 'f4':
            return differential_evolution(func, self.bounds, x0=self.x0, tol=1e-5, x0_init=self.x0)
        elif func.__name__ == 'f5':
            return differential_evolution(func, self.bounds, x0=self.x0, tol=1e-5, x0_init=self.x0, strategy='sbx')
        elif func.__name__ == 'f6':
            return differential_evolution(func, self.bounds, x0=self.x0, tol=1e-5, x0_init=self.x0, strategy='sbx', cxpb=0.5)
        elif func.__name__ == 'f7':
            return differential_evolution(func, self.bounds, x0=self.x0, tol=1e-5, x0_init=self.x0, strategy='sbx', cxpb=0.5, mu=0.5)
        elif func.__name__ == 'f8':
            return differential_evolution(func, self.bounds, x0=self.x0, tol=1e-5, x0_init=self.x0, strategy='sbx', cxpb=0.5, mu=0.5, w=0.5)
        elif func.__name__ == 'f9':
            return differential_evolution(func, self.bounds, x0=self.x0, tol=1e-5, x0_init=self.x0, strategy='sbx', cxpb=0.5, mu=0.5, w=0.5, strategy='rand1')
        elif func.__name__ == 'f10':
            return differential_evolution(func, self.bounds, x0=self.x0, tol=1e-5, x0_init=self.x0, strategy='sbx', cxpb=0.5, mu=0.5, w=0.5, strategy='rand1', scaling='uniform')
        elif func.__name__ == 'f11':
            return differential_evolution(func, self.bounds, x0=self.x0, tol=1e-5, x0_init=self.x0, strategy='sbx', cxpb=0.5, mu=0.5, w=0.5, strategy='rand1', scaling='log-uniform')
        elif func.__name__ == 'f12':
            return differential_evolution(func, self.bounds, x0=self.x0, tol=1e-5, x0_init=self.x0, strategy='sbx', cxpb=0.5, mu=0.5, w=0.5, strategy='rand1', scaling='log-uniform', scaling='uniform')
        elif func.__name__ == 'f13':
            return differential_evolution(func, self.bounds, x0=self.x0, tol=1e-5, x0_init=self.x0, strategy='sbx', cxpb=0.5, mu=0.5, w=0.5, strategy='rand1', scaling='uniform', scaling='log-uniform')
        elif func.__name__ == 'f14':
            return differential_evolution(func, self.bounds, x0=self.x0, tol=1e-5, x0_init=self.x0, strategy='sbx', cxpb=0.5, mu=0.5, w=0.5, strategy='rand1', scaling='uniform', scaling='log-uniform', scaling='uniform')
        elif func.__name__ == 'f15':
            return differential_evolution(func, self.bounds, x0=self.x0, tol=1e-5, x0_init=self.x0, strategy='sbx', cxpb=0.5, mu=0.5, w=0.5, strategy='rand1', scaling='uniform', scaling='log-uniform', scaling='uniform', scaling='uniform')
        elif func.__name__ == 'f16':
            return differential_evolution(func, self.bounds, x0=self.x0, tol=1e-5, x0_init=self.x0, strategy='sbx', cxpb=0.5, mu=0.5, w=0.5, strategy='rand1', scaling='uniform', scaling='log-uniform', scaling='uniform', scaling='uniform')
        elif func.__name__ == 'f17':
            return differential_evolution(func, self.bounds, x0=self.x0, tol=1e-5, x0_init=self.x0, strategy='sbx', cxpb=0.5, mu=0.5, w=0.5, strategy='rand1', scaling='uniform', scaling='log-uniform', scaling='uniform', scaling='uniform')
        elif func.__name__ == 'f18':
            return differential_evolution(func, self.bounds, x0=self.x0, tol=1e-5, x0_init=self.x0, strategy='sbx', cxpb=0.5, mu=0.5, w=0.5, strategy='rand1', scaling='uniform', scaling='log-uniform', scaling='uniform', scaling='uniform')
        elif func.__name__ == 'f19':
            return differential_evolution(func, self.bounds, x0=self.x0, tol=1e-5, x0_init=self.x0, strategy='sbx', cxpb=0.5, mu=0.5, w=0.5, strategy='rand1', scaling='uniform', scaling='log-uniform', scaling='uniform', scaling='uniform')
        elif func.__name__ == 'f20':
            return differential_evolution(func, self.bounds, x0=self.x0, tol=1e-5, x0_init=self.x0, strategy='sbx', cxpb=0.5, mu=0.5, w=0.5, strategy='rand1', scaling='uniform', scaling='log-uniform', scaling='uniform', scaling='uniform')
        elif func.__name__ == 'f21':
            return differential_evolution(func, self.bounds, x0=self.x0, tol=1e-5, x0_init=self.x0, strategy='sbx', cxpb=0.5, mu=0.5, w=0.5, strategy='rand1', scaling='uniform', scaling='log-uniform', scaling='uniform', scaling='uniform')
        elif func.__name__ == 'f22':
            return differential_evolution(func, self.bounds, x0=self.x0, tol=1e-5, x0_init=self.x0, strategy='sbx', cxpb=0.5, mu=0.5, w=0.5, strategy='rand1', scaling='uniform', scaling='log-uniform', scaling='uniform', scaling='uniform')
        elif func.__name__ == 'f23':
            return differential_evolution(func, self.bounds, x0=self.x0, tol=1e-5, x0_init=self.x0, strategy='sbx', cxpb=0.5, mu=0.5, w=0.5, strategy='rand1', scaling='uniform', scaling='log-uniform', scaling='uniform', scaling='uniform')
        elif func.__name__ == 'f24':
            return differential_evolution(func, self.bounds, x0=self.x0, tol=1e-5, x0_init=self.x0, strategy='sbx', cxpb=0.5, mu=0.5, w=0.5, strategy='rand1', scaling='uniform', scaling='log-uniform', scaling='uniform', scaling='uniform', scaling='uniform')

    def hybridize(self, func):
        import random
        import copy

        # Initialize the population
        population = [copy.deepcopy(self.x0) for _ in range(100)]

        # Initialize the best individual
        best_individual = copy.deepcopy(self.x0)

        # Initialize the best fitness
        best_fitness = np.inf

        # Main loop
        for i in range(self.budget):
            # Select the best individual
            best_individual = max(population, key=lambda x: func(x))

            # Evaluate the fitness
            best_fitness = func(best_individual)

            # Print the best individual and fitness
            print(f"Iteration {i+1}: Best Individual = {best_individual}, Best Fitness = {best_fitness}")

            # Refine the strategy
            if random.random() < 0.4:
                # Select a random individual
                individual = random.choice(population)

                # Refine the strategy
                individual = self.refine_strategy(individual, func)

                # Update the population
                population = [copy.deepcopy(individual) if individual == best_individual else copy.deepcopy(individual) for individual in population]

        # Return the best individual and fitness
        return best_individual, best_fitness

    def refine_strategy(self, individual, func):
        # Select a random dimension
        dim = random.randint(0, self.dim-1)

        # Select a random value for the dimension
        value = random.uniform(self.bounds[dim][0], self.bounds[dim][1])

        # Create a new individual
        new_individual = copy.deepcopy(individual)
        new_individual[dim] = value

        # Evaluate the fitness
        new_fitness = func(new_individual)

        # If the new individual has better fitness, return it
        if new_fitness < func(individual):
            return new_individual

        # Otherwise, return the original individual
        return individual

# Usage
import numpy as np
from scipy.optimize import differential_evolution

# Define the functions
def f1(x):
    return x[0]**2 + x[1]**2

def f2(x):
    return x[0]**2 + x[1]**2 + x[2]**2 + x[3]**2

def f3(x):
    return (x[0] - 2)**2 + (x[1] - 3)**2

def f4(x):
    return (x[0] - 2)**2 + (x[1] - 3)**2 + (x[2] - 4)**2 + (x[3] - 5)**2

def f5(x):
    return (x[0] - 2)**2 + (x[1] - 3)**2 + (x[2] - 4)**2 + (x[3] - 5)**2 + (x[4] - 6)**2

def f6(x):
    return (x[0] - 2)**2 + (x[1] - 3)**2 + (x[2] - 4)**2 + (x[3] - 5)**2 + (x[4] - 6)**2 + (x[5] - 7)**2

def f7(x):
    return (x[0] - 2)**2 + (x[1] - 3)**2 + (x[2] - 4)**2 + (x[3] - 5)**2 + (x[4] - 6)**2 + (x[5] - 7)**2 + (x[6] - 8)**2

def f8(x):
    return (x[0] - 2)**2 + (x[1] - 3)**2 + (x[2] - 4)**2 + (x[3] - 5)**2 + (x[4] - 6)**2 + (x[5] - 7)**2 + (x[6] - 8)**2 + (x[7] - 9)**2

def f9(x):
    return (x[0] - 2)**2 + (x[1] - 3)**2 + (x[2] - 4)**2 + (x[3] - 5)**2 + (x[4] - 6)**2 + (x[5] - 7)**2 + (x[6] - 8)**2 + (x[7] - 9)**2 + (x[8] - 10)**2

def f10(x):
    return (x[0] - 2)**2 + (x[1] - 3)**2 + (x[2] - 4)**2 + (x[3] - 5)**2 + (x[4] - 6)**2 + (x[5] - 7)**2 + (x[6] - 8)**2 + (x[7] - 9)**2 + (x[8] - 10)**2 + (x[9] - 11)**2

def f11(x):
    return (x[0] - 2)**2 + (x[1] - 3)**2 + (x[2] - 4)**2 + (x[3] - 5)**2 + (x[4] - 6)**2 + (x[5] - 7)**2 + (x[6] - 8)**2 + (x[7] - 9)**2 + (x[8] - 10)**2 + (x[9] - 11)**2 + (x[10] - 12)**2

def f12(x):
    return (x[0] - 2)**2 + (x[1] - 3)**2 + (x[2] - 4)**2 + (x[3] - 5)**2 + (x[4] - 6)**2 + (x[5] - 7)**2 + (x[6] - 8)**2 + (x[7] - 9)**2 + (x[8] - 10)**2 + (x[9] - 11)**2 + (x[10] - 12)**2 + (x[11] - 13)**2

def f13(x):
    return (x[0] - 2)**2 + (x[1] - 3)**2 + (x[2] - 4)**2 + (x[3] - 5)**2 + (x[4] - 6)**2 + (x[5] - 7)**2 + (x[6] - 8)**2 + (x[7] - 9)**2 + (x[8] - 10)**2 + (x[9] - 11)**2 + (x[10] - 12)**2 + (x[11] - 13)**2 + (x[12] - 14)**2

def f14(x):
    return (x[0] - 2)**2 + (x[1] - 3)**2 + (x[2] - 4)**2 + (x[3] - 5)**2 + (x[4] - 6)**2 + (x[5] - 7)**2 + (x[6] - 8)**2 + (x[7] - 9)**2 + (x[8] - 10)**2 + (x[9] - 11)**2 + (x[10] - 12)**2 + (x[11] - 13)**2 + (x[12] - 14)**2 + (x[13] - 15)**2

def f15(x):
    return (x[0] - 2)**2 + (x[1] - 3)**2 + (x[2] - 4)**2 + (x[3] - 5)**2 + (x[4] - 6)**2 + (x[5] - 7)**2 + (x[6] - 8)**2 + (x[7] - 9)**2 + (x[8] - 10)**2 + (x[9] - 11)**2 + (x[10] - 12)**2 + (x[11] - 13)**2 + (x[12] - 14)**2 + (x[13] - 15)**2 + (x[14] - 16)**2

def f16(x):
    return (x[0] - 2)**2 + (x[1] - 3)**2 + (x[2] - 4)**2 + (x[3] - 5)**2 + (x[4] - 6)**2 + (x[5] - 7)**2 + (x[6] - 8)**2 + (x[7] - 9)**2 + (x[8] - 10)**2 + (x[9] - 11)**2 + (x[10] - 12)**2 + (x[11] - 13)**2 + (x[12] - 14)**2 + (x[13] - 15)**2 + (x[14] - 16)**2 + (x[15] - 17)**2

def f17(x):
    return (x[0] - 2)**2 + (x[1] - 3)**2 + (x[2] - 4)**2 + (x[3] - 5)**2 + (x[4] - 6)**2 + (x[5] - 7)**2 + (x[6] - 8)**2 + (x[7] - 9)**2 + (x[8] - 10)**2 + (x[9] - 11)**2 + (x[10] - 12)**2 + (x[11] - 13)**2 + (x[12] - 14)**2 + (x[13] - 15)**2 + (x[14] - 16)**2 + (x[15] - 17)**2 + (x[16] - 18)**2

def f18(x):
    return (x[0] - 2)**2 + (x[1] - 3)**2 + (x[2] - 4)**2 + (x[3] - 5)**2 + (x[4] - 6)**2 + (x[5] - 7)**2 + (x[6] - 8)**2 + (x[7] - 9)**2 + (x[8] - 10)**2 + (x[9] - 11)**2 + (x[10] - 12)**2 + (x[11] - 13)**2 + (x[12] - 14)**2 + (x[13] - 15)**2 + (x[14] - 16)**2 + (x[15] - 17)**2 + (x[16] - 18)**2 + (x[17] - 19)**2

def f19(x):
    return (x[0] - 2)**2 + (x[1] - 3)**2 + (x[2] - 4)**2 + (x[3] - 5)**2 + (x[4] - 6)**2 + (x[5] - 7)**2 + (x[6] - 8)**2 + (x[7] - 9)**2 + (x[8] - 10)**2 + (x[9] - 11)**2 + (x[10] - 12)**2 + (x[11] - 13)**2 + (x[12] - 14)**2 + (x[13] - 15)**2 + (x[14] - 16)**2 + (x[15] - 17)**2 + (x[16] - 18)**2 + (x[17] - 19)**2 + (x[18] - 20)**2

def f20(x):
    return (x[0] - 2)**2 + (x[1] - 3)**2 + (x[2] - 4)**2 + (x[3] - 5)**2 + (x[4] - 6)**2 + (x[5] - 7)**2 + (x[6] - 8)**2 + (x[7] - 9)**2 + (x[8] - 10)**2 + (x[9] - 11)**2 + (x[10] - 12)**2 + (x[11] - 13)**2 + (x[12] - 14)**2 + (x[13] - 15)**2 + (x[14] - 16)**2 + (x[15] - 17)**2 + (x[16] - 18)**2 + (x[17] - 19)**2 + (x[18] - 20)**2 + (x[19] - 21)**2

def f21(x):
    return (x[0] - 2)**2 + (x[1] - 3)**2 + (x[2] - 4)**2 + (x[3] - 5)**2 + (x[4] - 6)**2 + (x[5] - 7)**2 + (x[6] - 8)**2 + (x[7] - 9)**2 + (x[8] - 10)**2 + (x[9] - 11)**2 + (x[10] - 12)**2 + (x[11] - 13)**2 + (x[12] - 14)**2 + (x[13] - 15)**2 + (x[14] - 16)**2 + (x[15] - 17)**2 + (x[16] - 18)**2 + (x[17] - 19)**2 + (x[18] - 20)**2 + (x[19] - 21)**2 + (x[20] - 22)**2

def f22(x):
    return (x[0] - 2)**2 + (x[1] - 3)**2 + (x[2] - 4)**2 + (x[3] - 5)**2 + (x[4] - 6)**2 + (x[5] - 7)**2 + (x[6] - 8)**2 + (x[7] - 9)**2 + (x[8] - 10)**2 + (x[9] - 11)**2 + (x[10] - 12)**2 + (x[11] - 13)**2 + (x[12] - 14)**2 + (x[13] - 15)**2 + (x[14] - 16)**2 + (x[15] - 17)**2 + (x[16] - 18)**2 + (x[17] - 19)**2 + (x[18] - 20)**2 + (x[19] - 21)**2 + (x[20] - 22)**2 + (x[21] - 23)**2

def f23(x):
    return (x[0] - 2)**2 + (x[1] - 3)**2 + (x[2] - 4)**2 + (x[3] - 5)**2 + (x[4] - 6)**2 + (x[5] - 7)**2 + (x[6] - 8)**2 + (x[7] - 9)**2 + (x[8] - 10)**2 + (x[9] - 11)**2 + (x[10] - 12)**2 + (x[11] - 13)**2 + (x[12] - 14)**2 + (x[13] - 15)**2 + (x[14] - 16)**2 + (x[15] - 17)**2 + (x[16] - 18)**2 + (x[17] - 19)**2 + (x[18] - 20)**2 + (x[19] - 21)**2 + (x[20] - 22)**2 + (x[21] - 23)**2 + (x[22] - 24)**2

def f24(x):
    return (x[0] - 2)**2 + (x[1] - 3)**2 + (x[2] - 4)**2 + (x[3] - 5)**2 + (x[4] - 6)**2 + (x[5] - 7)**2 + (x[6] - 8)**2 + (x[7] - 9)**2 + (x[8] - 10)**2 + (x[9] - 11)**2 + (x[10] - 12)**2 + (x[11] - 13)**2 + (x[12] - 14)**2 + (x[13] - 15)**2 + (x[14] - 16)**2 + (x[15] - 17)**2 + (x[16] - 18)**2 + (x[17] - 19)**2 + (x[18] - 20)**2 + (x[19] - 21)**2 + (x[20] - 22)**2 + (x[21] - 23)**2 + (x[22] - 24)**2 + (x[23] - 25)**2

# Usage
if __name__ == "__main__":
    # Initialize the HEACOMBBO
    heacombbo = HEACOMBBO(budget=50, dim=2)

    # Define the function
    def f(x):
        return x[0]**2 + x[1]**2

    # Optimize the function
    best_individual, best_fitness = heacombbo.hybridize(f)

    # Print the best individual and fitness
    print(f"Best Individual = {best_individual}, Best Fitness = {best_fitness}")