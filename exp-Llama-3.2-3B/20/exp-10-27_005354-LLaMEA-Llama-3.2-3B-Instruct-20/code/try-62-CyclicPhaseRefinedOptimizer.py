import numpy as np

class CyclicPhaseRefinedOptimizer:
    def __init__(self, budget, dim):
        """
        Initialize the Cyclic-Phase refined optimizer.

        Parameters:
        budget (int): Maximum number of function evaluations.
        dim (int): Dimensionality of the search space.
        """
        self.budget = budget
        self.dim = dim
        self.x = np.zeros((budget, dim))
        self.f_values = np.zeros(budget)
        self.phase = 0
        self.shift = 0
        self.probability = 0.2

    def __call__(self, func):
        """
        Optimize the given black box function.

        Parameters:
        func (callable): The black box function to optimize.

        Returns:
        float: The optimized value of the function.
        """
        for i in range(self.budget):
            # Generate a random initial point
            x = np.random.uniform(-5.0, 5.0, self.dim)

            # Evaluate the function at the initial point
            f_value = func(x)

            # Store the result
            self.x[i] = x
            self.f_values[i] = f_value

            # Update the phase and shift
            self.phase = (self.phase + 1) % 3
            if self.phase == 0:
                self.shift = 0
            elif self.phase == 1:
                self.shift = np.random.uniform(0, 1)
            else:
                self.shift = -np.random.uniform(0, 1)

            # Update the current point using probability-based line search
            if np.random.rand() < self.probability:
                x = x + self.shift
                f_value = func(x)
                if f_value < self.f_values[i]:
                    self.x[i] = x
                    self.f_values[i] = f_value

        # Return the optimized value
        return self.f_values[-1]

# Example usage:
def bbb1(x):
    return np.sum(x**2)

def bbb2(x):
    return np.sum(x**3)

def bbb3(x):
    return np.sum(x**4)

def bbb4(x):
    return np.sum(x**5)

def bbb5(x):
    return np.sum(x**6)

def bbb6(x):
    return np.sum(x**7)

def bbb7(x):
    return np.sum(x**8)

def bbb8(x):
    return np.sum(x**9)

def bbb9(x):
    return np.sum(x**10)

def bbb10(x):
    return np.sum(x**11)

def bbb11(x):
    return np.sum(x**12)

def bbb12(x):
    return np.sum(x**13)

def bbb13(x):
    return np.sum(x**14)

def bbb14(x):
    return np.sum(x**15)

def bbb15(x):
    return np.sum(x**16)

def bbb16(x):
    return np.sum(x**17)

def bbb17(x):
    return np.sum(x**18)

def bbb18(x):
    return np.sum(x**19)

def bbb19(x):
    return np.sum(x**20)

def bbb20(x):
    return np.sum(x**21)

def bbb21(x):
    return np.sum(x**22)

def bbb22(x):
    return np.sum(x**23)

def bbb23(x):
    return np.sum(x**24)

# Initialize the optimizer
optimizer = CyclicPhaseRefinedOptimizer(budget=100, dim=10)

# Optimize the black box functions
for func in [bbb1, bbb2, bbb3, bbb4, bbb5, bbb6, bbb7, bbb8, bbb9, bbb10, bbb11, bbb12, bbb13, bbb14, bbb15, bbb16, bbb17, bbb18, bbb19, bbb20, bbb21, bbb22, bbb23]:
    print(func.__name__, optimizer(func))