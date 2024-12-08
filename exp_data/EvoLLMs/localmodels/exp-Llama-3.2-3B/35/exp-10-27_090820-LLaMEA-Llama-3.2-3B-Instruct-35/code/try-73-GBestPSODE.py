import numpy as np
import random

class GBestPSODE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = 50
        self.w = 0.7298
        self.c1 = 1.49618
        self.c2 = 2.049912
        self.f = 0.5
        self.x = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        self.fval = np.inf
        self.best_x = np.inf

    def __call__(self, func):
        for _ in range(self.budget):
            # Evaluate the function at the current population
            fval = func(self.x)

            # Update the best solution
            if fval < self.fval:
                self.fval = fval
                self.best_x = self.x[np.argmin(fval)]
                self.x = np.array([self.best_x])

            # Refine the population with probabilistic strategy
            refine_prob = 0.35
            refine_mask = np.random.rand(self.population_size) < refine_prob
            self.x = np.vstack((self.x, np.array([random.uniform(self.lower_bound, self.upper_bound) for _ in range(self.dim)])))
            self.x = self.x[~refine_mask]
            self.x = self.x[np.argsort(np.abs(self.x - self.best_x))]
            self.x = self.x[:self.population_size]

            # Apply PSO and DE operators
            v = self.w * np.random.uniform(0, 1, (self.population_size, self.dim)) + self.c1 * np.abs(self.x - self.best_x[:, np.newaxis]) + self.c2 * np.abs(self.x - np.mean(self.x, axis=0)[:, np.newaxis]) ** self.f
            self.x = self.x + v

            # Limit the search space
            self.x = np.clip(self.x, self.lower_bound, self.upper_bound)

            # Evaluate the function at the updated population
            fval = func(self.x)

            # Update the best solution
            if fval < self.fval:
                self.fval = fval
                self.best_x = self.x[np.argmin(fval)]

            # Select the best individual
            self.x = self.x[np.argmin(fval)]

        return self.fval, self.best_x

# Example usage:
def bbb_function1(x):
    return x[0]**2 + x[1]**2 + x[2]**2

def bbb_function2(x):
    return x[0]**2 + x[1]**2 - x[2]**2

def bbb_function3(x):
    return x[0]**2 + x[1]**2 + x[2]**2 + x[3]**2

def bbb_function4(x):
    return x[0]**2 + x[1]**2 - x[2]**2 - x[3]**2

def bbb_function5(x):
    return x[0]**2 + x[1]**2 + x[2]**2 + x[3]**2 + x[4]**2

def bbb_function6(x):
    return x[0]**2 + x[1]**2 - x[2]**2 - x[3]**2 - x[4]**2

def bbb_function7(x):
    return x[0]**2 + x[1]**2 + x[2]**2 + x[3]**2 + x[4]**2 + x[5]**2

def bbb_function8(x):
    return x[0]**2 + x[1]**2 - x[2]**2 - x[3]**2 - x[4]**2 - x[5]**2

def bbb_function9(x):
    return x[0]**2 + x[1]**2 + x[2]**2 + x[3]**2 + x[4]**2 + x[5]**2 + x[6]**2

def bbb_function10(x):
    return x[0]**2 + x[1]**2 - x[2]**2 - x[3]**2 - x[4]**2 - x[5]**2 - x[6]**2

def bbb_function11(x):
    return x[0]**2 + x[1]**2 + x[2]**2 + x[3]**2 + x[4]**2 + x[5]**2 + x[6]**2 + x[7]**2

def bbb_function12(x):
    return x[0]**2 + x[1]**2 - x[2]**2 - x[3]**2 - x[4]**2 - x[5]**2 - x[6]**2 - x[7]**2

def bbb_function13(x):
    return x[0]**2 + x[1]**2 + x[2]**2 + x[3]**2 + x[4]**2 + x[5]**2 + x[6]**2 + x[7]**2 + x[8]**2

def bbb_function14(x):
    return x[0]**2 + x[1]**2 - x[2]**2 - x[3]**2 - x[4]**2 - x[5]**2 - x[6]**2 - x[7]**2 - x[8]**2

def bbb_function15(x):
    return x[0]**2 + x[1]**2 + x[2]**2 + x[3]**2 + x[4]**2 + x[5]**2 + x[6]**2 + x[7]**2 + x[8]**2 + x[9]**2

def bbb_function16(x):
    return x[0]**2 + x[1]**2 - x[2]**2 - x[3]**2 - x[4]**2 - x[5]**2 - x[6]**2 - x[7]**2 - x[8]**2 - x[9]**2

def bbb_function17(x):
    return x[0]**2 + x[1]**2 + x[2]**2 + x[3]**2 + x[4]**2 + x[5]**2 + x[6]**2 + x[7]**2 + x[8]**2 + x[9]**2 + x[10]**2

def bbb_function18(x):
    return x[0]**2 + x[1]**2 - x[2]**2 - x[3]**2 - x[4]**2 - x[5]**2 - x[6]**2 - x[7]**2 - x[8]**2 - x[9]**2 - x[10]**2

def bbb_function19(x):
    return x[0]**2 + x[1]**2 + x[2]**2 + x[3]**2 + x[4]**2 + x[5]**2 + x[6]**2 + x[7]**2 + x[8]**2 + x[9]**2 + x[10]**2 + x[11]**2

def bbb_function20(x):
    return x[0]**2 + x[1]**2 - x[2]**2 - x[3]**2 - x[4]**2 - x[5]**2 - x[6]**2 - x[7]**2 - x[8]**2 - x[9]**2 - x[10]**2 - x[11]**2

def bbb_function21(x):
    return x[0]**2 + x[1]**2 + x[2]**2 + x[3]**2 + x[4]**2 + x[5]**2 + x[6]**2 + x[7]**2 + x[8]**2 + x[9]**2 + x[10]**2 + x[11]**2 + x[12]**2

def bbb_function22(x):
    return x[0]**2 + x[1]**2 - x[2]**2 - x[3]**2 - x[4]**2 - x[5]**2 - x[6]**2 - x[7]**2 - x[8]**2 - x[9]**2 - x[10]**2 - x[11]**2 - x[12]**2

def bbb_function23(x):
    return x[0]**2 + x[1]**2 + x[2]**2 + x[3]**2 + x[4]**2 + x[5]**2 + x[6]**2 + x[7]**2 + x[8]**2 + x[9]**2 + x[10]**2 + x[11]**2 + x[12]**2 + x[13]**2

def bbb_function24(x):
    return x[0]**2 + x[1]**2 - x[2]**2 - x[3]**2 - x[4]**2 - x[5]**2 - x[6]**2 - x[7]**2 - x[8]**2 - x[9]**2 - x[10]**2 - x[11]**2 - x[12]**2 - x[13]**2

# Example usage:
algorithm = GBestPSODE(100, 2)
best_x, best_f = algorithm(bbb_function1)
print(f"Best x: {best_x}, Best f: {best_f}")

best_x, best_f = algorithm(bbb_function2)
print(f"Best x: {best_x}, Best f: {best_f}")

best_x, best_f = algorithm(bbb_function3)
print(f"Best x: {best_x}, Best f: {best_f}")

best_x, best_f = algorithm(bbb_function4)
print(f"Best x: {best_x}, Best f: {best_f}")

best_x, best_f = algorithm(bbb_function5)
print(f"Best x: {best_x}, Best f: {best_f}")

best_x, best_f = algorithm(bbb_function6)
print(f"Best x: {best_x}, Best f: {best_f}")

best_x, best_f = algorithm(bbb_function7)
print(f"Best x: {best_x}, Best f: {best_f}")

best_x, best_f = algorithm(bbb_function8)
print(f"Best x: {best_x}, Best f: {best_f}")

best_x, best_f = algorithm(bbb_function9)
print(f"Best x: {best_x}, Best f: {best_f}")

best_x, best_f = algorithm(bbb_function10)
print(f"Best x: {best_x}, Best f: {best_f}")

best_x, best_f = algorithm(bbb_function11)
print(f"Best x: {best_x}, Best f: {best_f}")

best_x, best_f = algorithm(bbb_function12)
print(f"Best x: {best_x}, Best f: {best_f}")

best_x, best_f = algorithm(bbb_function13)
print(f"Best x: {best_x}, Best f: {best_f}")

best_x, best_f = algorithm(bbb_function14)
print(f"Best x: {best_x}, Best f: {best_f}")

best_x, best_f = algorithm(bbb_function15)
print(f"Best x: {best_x}, Best f: {best_f}")

best_x, best_f = algorithm(bbb_function16)
print(f"Best x: {best_x}, Best f: {best_f}")

best_x, best_f = algorithm(bbb_function17)
print(f"Best x: {best_x}, Best f: {best_f}")

best_x, best_f = algorithm(bbb_function18)
print(f"Best x: {best_x}, Best f: {best_f}")

best_x, best_f = algorithm(bbb_function19)
print(f"Best x: {best_x}, Best f: {best_f}")

best_x, best_f = algorithm(bbb_function20)
print(f"Best x: {best_x}, Best f: {best_f}")

best_x, best_f = algorithm(bbb_function21)
print(f"Best x: {best_x}, Best f: {best_f}")

best_x, best_f = algorithm(bbb_function22)
print(f"Best x: {best_x}, Best f: {best_f}")

best_x, best_f = algorithm(bbb_function23)
print(f"Best x: {best_x}, Best f: {best_f}")

best_x, best_f = algorithm(bbb_function24)
print(f"Best x: {best_x}, Best f: {best_f}")