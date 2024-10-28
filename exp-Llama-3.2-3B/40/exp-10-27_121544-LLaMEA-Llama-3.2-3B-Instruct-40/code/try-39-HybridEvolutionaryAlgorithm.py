import numpy as np
from scipy.optimize import differential_evolution

class HybridEvolutionaryAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.bounds = [(-5.0, 5.0)] * dim
        self.x0 = np.random.uniform(self.bounds[0][0], self.bounds[0][1], self.dim)

    def __call__(self, func):
        if self.budget == 0:
            return np.nan, np.nan

        for _ in range(self.budget):
            new_individual = np.random.uniform(self.bounds[0][0], self.bounds[0][1], self.dim)
            if np.random.rand() < 0.4:
                new_individual = self.f(new_individual, func)
            updated_individual = self.f(new_individual, func)
            if updated_individual[0] < self.x0[0]:
                self.x0 = updated_individual
        return self.x0, self.f(self.x0, func)

    def f(self, individual, func):
        if np.random.rand() < 0.4:
            individual = self.crossover(individual)
            individual = self.mutation(individual)
        return func(individual)

    def crossover(self, individual):
        new_individual = np.random.uniform(self.bounds[0][0], self.bounds[0][1], self.dim)
        new_individual[1:] = individual[1:]
        return new_individual

    def mutation(self, individual):
        new_individual = individual.copy()
        for i in range(self.dim):
            if np.random.rand() < 0.1:
                new_individual[i] = np.random.uniform(self.bounds[0][0], self.bounds[0][1])
        return new_individual

# Example usage:
class BBOB:
    def __init__(self, func):
        self.func = func

    def f(self, x):
        return self.func(x)

# Test the HybridEvolutionaryAlgorithm
def f1(x):
    return np.sum(x**2)

def f2(x):
    return np.sum(x**2) + 10 * np.sin(2 * np.pi * x)

def f3(x):
    return np.sum(x**2) + 10 * np.sin(2 * np.pi * x) + 5 * x[0]

def f4(x):
    return np.sum(x**2) + 10 * np.sin(2 * np.pi * x) + 5 * x[0] + 5 * x[1]

def f5(x):
    return np.sum(x**2) + 10 * np.sin(2 * np.pi * x) + 5 * x[0] + 5 * x[1] + 2 * x[0] * x[1]

def f6(x):
    return np.sum(x**2) + 10 * np.sin(2 * np.pi * x) + 5 * x[0] + 5 * x[1] + 2 * x[0] * x[1] + 0.5 * x[0] * x[1] * x[0] * x[1]

def f7(x):
    return np.sum(x**2) + 10 * np.sin(2 * np.pi * x) + 5 * x[0] + 5 * x[1] + 2 * x[0] * x[1] + 0.5 * x[0] * x[1] * x[0] * x[1] + 0.5 * x[0] * x[1] * x[0] * x[1] * x[0] * x[1]

def f8(x):
    return np.sum(x**2) + 10 * np.sin(2 * np.pi * x) + 5 * x[0] + 5 * x[1] + 2 * x[0] * x[1] + 0.5 * x[0] * x[1] * x[0] * x[1] + 0.5 * x[0] * x[1] * x[0] * x[1] * x[0] * x[1] + 0.5 * x[0] * x[1] * x[0] * x[1] * x[0] * x[1] * x[0] * x[1]

def f9(x):
    return np.sum(x**2) + 10 * np.sin(2 * np.pi * x) + 5 * x[0] + 5 * x[1] + 2 * x[0] * x[1] + 0.5 * x[0] * x[1] * x[0] * x[1] + 0.5 * x[0] * x[1] * x[0] * x[1] * x[0] * x[1] + 0.5 * x[0] * x[1] * x[0] * x[1] * x[0] * x[1] * x[0] * x[1] + 0.5 * x[0] * x[1] * x[0] * x[1] * x[0] * x[1] * x[0] * x[1] * x[0] * x[1]

def f10(x):
    return np.sum(x**2) + 10 * np.sin(2 * np.pi * x) + 5 * x[0] + 5 * x[1] + 2 * x[0] * x[1] + 0.5 * x[0] * x[1] * x[0] * x[1] + 0.5 * x[0] * x[1] * x[0] * x[1] * x[0] * x[1] + 0.5 * x[0] * x[1] * x[0] * x[1] * x[0] * x[1] * x[0] * x[1] + 0.5 * x[0] * x[1] * x[0] * x[1] * x[0] * x[1] * x[0] * x[1] * x[0] * x[1] + 0.5 * x[0] * x[1] * x[0] * x[1] * x[0] * x[1] * x[0] * x[1] * x[0] * x[1] * x[0] * x[1]

def f11(x):
    return np.sum(x**2) + 10 * np.sin(2 * np.pi * x) + 5 * x[0] + 5 * x[1] + 2 * x[0] * x[1] + 0.5 * x[0] * x[1] * x[0] * x[1] + 0.5 * x[0] * x[1] * x[0] * x[1] * x[0] * x[1] + 0.5 * x[0] * x[1] * x[0] * x[1] * x[0] * x[1] * x[0] * x[1] + 0.5 * x[0] * x[1] * x[0] * x[1] * x[0] * x[1] * x[0] * x[1] * x[0] * x[1] + 0.5 * x[0] * x[1] * x[0] * x[1] * x[0] * x[1] * x[0] * x[1] * x[0] * x[1] * x[0] * x[1] + 0.5 * x[0] * x[1] * x[0] * x[1] * x[0] * x[1] * x[0] * x[1] * x[0] * x[1] * x[0] * x[1] * x[0] * x[1]

def f12(x):
    return np.sum(x**2) + 10 * np.sin(2 * np.pi * x) + 5 * x[0] + 5 * x[1] + 2 * x[0] * x[1] + 0.5 * x[0] * x[1] * x[0] * x[1] + 0.5 * x[0] * x[1] * x[0] * x[1] * x[0] * x[1] + 0.5 * x[0] * x[1] * x[0] * x[1] * x[0] * x[1] * x[0] * x[1] + 0.5 * x[0] * x[1] * x[0] * x[1] * x[0] * x[1] * x[0] * x[1] * x[0] * x[1] + 0.5 * x[0] * x[1] * x[0] * x[1] * x[0] * x[1] * x[0] * x[1] * x[0] * x[1] * x[0] * x[1] + 0.5 * x[0] * x[1] * x[0] * x[1] * x[0] * x[1] * x[0] * x[1] * x[0] * x[1] * x[0] * x[1] * x[0] * x[1]

def f13(x):
    return np.sum(x**2) + 10 * np.sin(2 * np.pi * x) + 5 * x[0] + 5 * x[1] + 2 * x[0] * x[1] + 0.5 * x[0] * x[1] * x[0] * x[1] + 0.5 * x[0] * x[1] * x[0] * x[1] * x[0] * x[1] + 0.5 * x[0] * x[1] * x[0] * x[1] * x[0] * x[1] * x[0] * x[1] + 0.5 * x[0] * x[1] * x[0] * x[1] * x[0] * x[1] * x[0] * x[1] * x[0] * x[1] + 0.5 * x[0] * x[1] * x[0] * x[1] * x[0] * x[1] * x[0] * x[1] * x[0] * x[1] * x[0] * x[1] + 0.5 * x[0] * x[1] * x[0] * x[1] * x[0] * x[1] * x[0] * x[1] * x[0] * x[1] * x[0] * x[1] * x[0] * x[1]

def f14(x):
    return np.sum(x**2) + 10 * np.sin(2 * np.pi * x) + 5 * x[0] + 5 * x[1] + 2 * x[0] * x[1] + 0.5 * x[0] * x[1] * x[0] * x[1] + 0.5 * x[0] * x[1] * x[0] * x[1] * x[0] * x[1] + 0.5 * x[0] * x[1] * x[0] * x[1] * x[0] * x[1] * x[0] * x[1] + 0.5 * x[0] * x[1] * x[0] * x[1] * x[0] * x[1] * x[0] * x[1] * x[0] * x[1] + 0.5 * x[0] * x[1] * x[0] * x[1] * x[0] * x[1] * x[0] * x[1] * x[0] * x[1] * x[0] * x[1] + 0.5 * x[0] * x[1] * x[0] * x[1] * x[0] * x[1] * x[0] * x[1] * x[0] * x[1] * x[0] * x[1] * x[0] * x[1]

def f15(x):
    return np.sum(x**2) + 10 * np.sin(2 * np.pi * x) + 5 * x[0] + 5 * x[1] + 2 * x[0] * x[1] + 0.5 * x[0] * x[1] * x[0] * x[1] + 0.5 * x[0] * x[1] * x[0] * x[1] * x[0] * x[1] + 0.5 * x[0] * x[1] * x[0] * x[1] * x[0] * x[1] * x[0] * x[1] + 0.5 * x[0] * x[1] * x[0] * x[1] * x[0] * x[1] * x[0] * x[1] * x[0] * x[1] + 0.5 * x[0] * x[1] * x[0] * x[1] * x[0] * x[1] * x[0] * x[1] * x[0] * x[1] * x[0] * x[1] + 0.5 * x[0] * x[1] * x[0] * x[1] * x[0] * x[1] * x[0] * x[1] * x[0] * x[1] * x[0] * x[1] * x[0] * x[1]

def f16(x):
    return np.sum(x**2) + 10 * np.sin(2 * np.pi * x) + 5 * x[0] + 5 * x[1] + 2 * x[0] * x[1] + 0.5 * x[0] * x[1] * x[0] * x[1] + 0.5 * x[0] * x[1] * x[0] * x[1] * x[0] * x[1] + 0.5 * x[0] * x[1] * x[0] * x[1] * x[0] * x[1] * x[0] * x[1] + 0.5 * x[0] * x[1] * x[0] * x[1] * x[0] * x[1] * x[0] * x[1] * x[0] * x[1] + 0.5 * x[0] * x[1] * x[0] * x[1] * x[0] * x[1] * x[0] * x[1] * x[0] * x[1] * x[0] * x[1] + 0.5 * x[0] * x[1] * x[0] * x[1] * x[0] * x[1] * x[0] * x[1] * x[0] * x[1] * x[0] * x[1] * x[0] * x[1]

def f17(x):
    return np.sum(x**2) + 10 * np.sin(2 * np.pi * x) + 5 * x[0] + 5 * x[1] + 2 * x[0] * x[1] + 0.5 * x[0] * x[1] * x[0] * x[1] + 0.5 * x[0] * x[1] * x[0] * x[1] * x[0] * x[1] + 0.5 * x[0] * x[1] * x[0] * x[1] * x[0] * x[1] * x[0] * x[1] + 0.5 * x[0] * x[1] * x[0] * x[1] * x[0] * x[1] * x[0] * x[1] * x[0] * x[1] + 0.5 * x[0] * x[1] * x[0] * x[1] * x[0] * x[1] * x[0] * x[1] * x[0] * x[1] * x[0] * x[1] + 0.5 * x[0] * x[1] * x[0] * x[1] * x[0] * x[1] * x[0] * x[1] * x[0] * x[1] * x[0] * x[1] * x[0] * x[1]

def f18(x):
    return np.sum(x**2) + 10 * np.sin(2 * np.pi * x) + 5 * x[0] + 5 * x[1] + 2 * x[0] * x[1] + 0.5 * x[0] * x[1] * x[0] * x[1] + 0.5 * x[0] * x[1] * x[0] * x[1] * x[0] * x[1] + 0.5 * x[0] * x[1] * x[0] * x[1] * x[0] * x[1] * x[0] * x[1] + 0.5 * x[0] * x[1] * x[0] * x[1] * x[0] * x[1] * x[0] * x[1] * x[0] * x[1] + 0.5 * x[0] * x[1] * x[0] * x[1] * x[0] * x[1] * x[0] * x[1] * x[0] * x[1] * x[0] * x[1] + 0.5 * x[0] * x[1] * x[0] * x[1] * x[0] * x[1] * x[0] * x[1] * x[0] * x[1] * x[0] * x[1] * x[0] * x[1]

def f19(x):
    return np.sum(x**2) + 10 * np.sin(2 * np.pi * x) + 5 * x[0] + 5 * x[1] + 2 * x[0] * x[1] + 0.5 * x[0] * x[1] * x[0] * x[1] + 0.5 * x[0] * x[1] * x[0] * x[1] * x[0] * x[1] + 0.5 * x[0] * x[1] * x[0] * x[1] * x[0] * x[1] * x[0] * x[1] + 0.5 * x[0] * x[1] * x[0] * x[1] * x[0] * x[1] * x[0] * x[1] * x[0] * x[1] + 0.5 * x[0] * x[1] * x[0] * x[1] * x[0] * x[1] * x[0] * x[1] * x[0] * x[1] * x[0] * x[1] + 0.5 * x[0] * x[1] * x[0] * x[1] * x[0] * x[1] * x[0] * x[1] * x[0] * x[1] * x[0] * x[1] * x[0] * x[1]

def f20(x):
    return np.sum(x**2) + 10 * np.sin(2 * np.pi * x) + 5 * x[0] + 5 * x[1] + 2 * x[0] * x[1] + 0.5 * x[0] * x[1] * x[0] * x[1] + 0.5 * x[0] * x[1] * x[0] * x[1] * x[0] * x[1] + 0.5 * x[0] * x[1] * x[0] * x[1] * x[0] * x[1] * x[0] * x[1] + 0.5 * x[0] * x[1] * x[0] * x[1] * x[0] * x[1] * x[0] * x[1] * x[0] * x[1] + 0.5 * x[0] * x[1] * x[0] * x[1] * x[0] * x[1] * x[0] * x[1] * x[0] * x[1] * x[0] * x[1] + 0.5 * x[0] * x[1] * x[0] * x[1] * x[0] * x[1] * x[0] * x[1] * x[0] * x[1] * x[0] * x[1] * x[0] * x[1]

def f21(x):
    return np.sum(x**2) + 10 * np.sin(2 * np.pi * x) + 5 * x[0] + 5 * x[1] + 2 * x[0] * x[1] + 0.5 * x[0] * x[1] * x[0] * x[1] + 0.5 * x[0] * x[1] * x[0] * x[1] * x[0] * x[1] + 0.5 * x[0] * x[1] * x[0] * x[1] * x[0] * x[1] * x[0] * x[1] + 0.5 * x[0] * x[1] * x[0] * x[1] * x[0] * x[1] * x[0] * x[1] * x[0] * x[1] + 0.5 * x[0] * x[1] * x[0] * x[1] * x[0] * x[1] * x[0] * x[1] * x[0] * x[1] * x[0] * x[1] + 0.5 * x[0] * x[1] * x[0] * x[1] * x[0] * x[1] * x[0] * x[1] * x[0] * x[1] * x[0] * x[1] * x[0] * x[1]

def f22(x):
    return np.sum(x**2) + 10 * np.sin(2 * np.pi * x) + 5 * x[0] + 5 * x[1] + 2 * x[0] * x[1] + 0.5 * x[0] * x[1] * x[0] * x[1] + 0.5 * x[0] * x[1] * x[0] * x[1] * x[0] * x[1] + 0.5 * x[0] * x[1] * x[0] * x[1] * x[0] * x[1] * x[0] * x[1] + 0.5 * x[0] * x[1] * x[0] * x[1] * x[0] * x[1] * x[0] * x[1] * x[0] * x[1] + 0.5 * x[0] * x[1] * x[0] * x[1] * x[0] * x[1] * x[0] * x[1] * x[0] * x[1] * x[0] * x[1] + 0.5 * x[0] * x[1] * x[0] * x[1] * x[0] * x[1] * x[0] * x[1] * x[0] * x[1] * x[0] * x[1] * x[0] * x[1]

def f23(x):
    return np.sum(x**2) + 10 * np.sin(2 * np.pi * x) + 5 * x[0] + 5 * x[1] + 2 * x[0] * x[1] + 0.5 * x[0] * x[1] * x[0] * x[1] + 0.5 * x[0] * x[1] * x[0] * x[1] * x[0] * x[1] + 0.5 * x[0] * x[1] * x[0] * x[1] * x[0] * x[1] * x[0] * x[1] + 0.5 * x[0] * x[1] * x[0] * x[1] * x[0] * x[1] * x[0] * x[1] * x[0] * x[1] + 0.5 * x[0] * x[1] * x[0] * x[1] * x[0] * x[1] * x[0] * x[1] * x[0] * x[1] * x[0] * x[1] + 0.5 * x[0] * x[1] * x[0] * x[1] * x[0] * x[1] * x[0] * x[1] * x[0] * x[1] * x[0] * x[1] * x[0] * x[1]

def f24(x):
    return np.sum(x**2) + 10 * np.sin(2 * np.pi * x) + 5 * x[0] + 5 * x[1] + 2 * x[0] * x[1] + 0.5 * x[0] * x[1] * x[0] * x[1] + 0.5 * x[0] * x[1] * x[0] * x[1] * x[0] * x[1] + 0.5 * x[0] * x[1] * x[0] * x[1] * x[0] * x[1] * x[0] * x[1] + 0.5 * x[0] * x[1] * x[0] * x[1] * x[0] * x[1] * x[0] * x[1] * x[0] * x[1] + 0.5 * x[0] * x[1] * x[0] * x[1] * x[0] * x[1] * x[0] * x[1] * x[0] * x[1] * x[0] * x[1] + 0.5 * x[0] * x[1] * x[0] * x[1] * x[0] * x[1] * x[0] * x[1] * x[0] * x[1] * x[0] * x[1] * x[0] * x[1]

# Initialize the HybridEvolutionaryAlgorithm
heacombbo = HybridEvolutionaryAlgorithm(budget=10, dim=10)

# Test the HybridEvolutionaryAlgorithm
bbo = BBO(f1)
x0, f1_x0 = heacombbo(bbo)
print("x0:", x0)
print("f1(x0):", f1_x0)

bbo = BBO(f2)
x0, f2_x0 = heacombbo(bbo)
print("x0:", x0)
print("f2(x0):", f2_x0)

bbo = BBO(f3)
x0, f3_x0 = heacombbo(bbo)
print("x0:", x0)
print("f3(x0):", f3_x0)

bbo = BBO(f4)
x0, f4_x0 = heacombbo(bbo)
print("x0:", x0)
print("f4(x0):", f4_x0)

bbo = BBO(f5)
x0, f5_x0 = heacombbo(bbo)
print("x0:", x0)
print("f5(x0):", f5_x0)

bbo = BBO(f6)
x0, f6_x0 = heacombbo(bbo)
print("x0:", x0)
print("f6(x0):", f6_x0)

bbo = BBO(f7)
x0, f7_x0 = heacombbo(bbo)
print("x0:", x0)
print("f7(x0):", f7_x0)

bbo = BBO(f8)
x0, f8_x0 = heacombbo(bbo)
print("x0:", x0)
print("f8(x0):", f8_x0)

bbo = BBO(f9)
x0, f9_x0 = heacombbo(bbo)
print("x0:", x0)
print("f9(x0):", f9_x0)

bbo = BBO(f10)
x0, f10_x0 = heacombbo(bbo)
print("x0:", x0)
print("f10(x0):", f10_x0)

bbo = BBO(f11)
x0, f11_x0 = heacombbo(bbo)
print("x0:", x0)
print("f11(x0):", f11_x0)

bbo = BBO(f12)
x0, f12_x0 = heacombbo(bbo)
print("x0:", x0)
print("f12(x0):", f12_x0)

bbo = BBO(f13)
x0, f13_x0 = heacombbo(bbo)
print("x0:", x0)
print("f13(x0):", f13_x0)

bbo = BBO(f14)
x0, f14_x0 = heacombbo(bbo)
print("x0:", x0)
print("f14(x0):", f14_x0)

bbo = BBO(f15)
x0, f15_x0 = heacombbo(bbo)
print("x0:", x0)
print("f15(x0):", f15_x0)

bbo = BBO(f16)
x0, f16_x0 = heacombbo(bbo)
print("x0:", x0)
print("f16(x0):", f16_x0)

bbo = BBO(f17)
x0, f17_x0 = heacombbo(bbo)
print("x0:", x0)
print("f17(x0):", f17_x0)

bbo = BBO(f18)
x0, f18_x0 = heacombbo(bbo)
print("x0:", x0)
print("f18(x0):", f18_x0)

bbo = BBO(f19)
x0, f19_x0 = heacombbo(bbo)
print("x0:", x0)
print("f19(x0):", f19_x0)

bbo = BBO(f20)
x0, f20_x0 = heacombbo(bbo)
print("x0:", x0)
print("f20(x0):", f20_x0)

bbo = BBO(f21)
x0, f21_x0 = heacombbo(bbo)
print("x0:", x0)
print("f21(x0):", f21_x0)

bbo = BBO(f22)
x0, f22_x0 = heacombbo(bbo)
print("x0:", x0)
print("f22(x0):", f22_x0)

bbo = BBO(f23)
x0, f23_x0 = heacombbo(bbo)
print("x0:", x0)
print("f23(x0):", f23_x0)

bbo = BBO(f24)
x0, f24_x0 = heacombbo(bbo)
print("x0:", x0)
print("f24(x0):", f24_x0)