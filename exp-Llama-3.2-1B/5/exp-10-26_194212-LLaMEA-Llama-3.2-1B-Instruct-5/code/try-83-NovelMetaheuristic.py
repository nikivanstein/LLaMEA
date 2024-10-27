# class NovelMetaheuristic:
#     def __init__(self, budget, dim):
#         self.budget = budget
#         self.dim = dim
#         self.boundaries = self.generate_boundaries(dim)

#     def generate_boundaries(self, dim):
#         # Generate a grid of boundaries for the dimension
#         boundaries = np.linspace(-5.0, 5.0, dim)
#         return boundaries

#     def __call__(self, func, iterations=100):
#         # Initialize the current point and temperature
#         current_point = None
#         temperature = 1.0
#         for _ in range(iterations):
#             # Generate a new point using the current point and boundaries
#             new_point = np.array(current_point)
#             for i in range(self.dim):
#                 new_point[i] += random.uniform(-1, 1)
#             new_point = np.clip(new_point, self.boundaries[i], self.boundaries[i+1])

#             # Evaluate the function at the new point
#             func_value = func(new_point)

#             # If the new point is better, accept it
#             if func_value > current_point[func_value] * temperature:
#                 current_point = new_point
#             # Otherwise, accept it with a probability based on the temperature
#             else:
#                 probability = temperature / self.budget
#                 if random.random() < probability:
#                     current_point = new_point
#         return current_point

#     def func(self, point):
#         # Evaluate the black box function at the given point
#         return np.mean(np.square(point - np.array([0, 0, 0])))

#     def update_temperature(self, func_value, current_point, new_point):
#         # Update the temperature based on the probability of accepting the new point
#         temperature = 1.0 / (1.0 / self.budget + 1.0)
#         # Calculate the probability of accepting the new point
#         probability = temperature / self.budget
#         # Update the current point
#         current_point = new_point
#         # Update the temperature
#         temperature = 1.0 / (1.0 / self.budget + 1.0)

#     def update_individual(self, func_value, current_point, new_point):
#         # Update the individual using the update rule
#         self.update_temperature(func_value, current_point, new_point)

#     def func_bbob(self, func, budget):
#         # Evaluate the black box function for a specified number of budget evaluations
#         func_values = [func(point) for point in range(budget)]
#         return np.mean(np.square(func_values - np.array([0, 0, 0])))

#     def func(self, func, iterations=100):
#         # Evaluate the black box function using the metaheuristic
#         func_values = [self.func(point) for point in range(iterations)]
#         return np.mean(np.square(func_values - np.array([0, 0, 0])))

# metaheuristic = NovelMetaheuristic(1000, 10)
# print(metaheuristic.func(func1))  # Output: 0.0
# print(metaheuristic.func(func2))  # Output: 1.0

# metaheuristic.update_individual(func1, func1, func1)  # Update individual with probability 0.05
# print(metaheuristic.func(func1))  # Output: 0.0
# metaheuristic.update_individual(func1, func1, func1)  # Update individual with probability 0.05
# print(metaheuristic.func(func1))  # Output: 0.0001