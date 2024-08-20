"""LLaMEA - LLM powered Evolutionary Algorithm for code optimization
This module integrates OpenAI's language models to generate and evolve
algorithms to automatically evaluate (for example metaheuristics evaluated on BBOB).
"""
import json
import re
import traceback

import numpy as np
from ConfigSpace import ConfigurationSpace

from .llm import LLMmanager
from .loggers import ExperimentLogger
from .utils import NoCodeException


class LLaMEA:
    """
    A class that represents the Language Model powered Evolutionary Algorithm (LLaMEA).
    This class handles the initialization, evolution, and interaction with a language model
    to generate and refine algorithms.
    """

    def __init__(
        self,
        f,
        api_key,
        role_prompt="",
        task_prompt="",
        experiment_name="",
        elitism=False,
        HPO=False,
        feedback_prompt="",
        budget=100,
        model="gpt-4-turbo",
        log=True,
    ):
        """
        Initializes the LLaMEA instance with provided parameters.

        Args:
            f (callable): The evaluation function to measure the fitness of algorithms.
            api_key (str): The API key for accessing OpenAI's services.
            role_prompt (str): A prompt that defines the role of the language model in the optimization task.
            task_prompt (str): A prompt describing the task for the language model to generate optimization algorithms.
            experiment_name (str): The name of the experiment for logging purposes.
            elitism (bool): Flag to decide if elitism should be used in the evolutionary process.
            HPO (bool): Flag to decide if hyper-parameter optimization is part of the evaluation function.
                In case it is, a configuration space should be asked from the LLM as additional output in json format.
            feedback_prompt (str): Prompt to guide the model on how to provide feedback on the generated algorithms.
            budget (int): The number of generations to run the evolutionary algorithm.
            model (str): The model identifier from OpenAI or ollama to be used.
            log (bool): Flag to switch of the logging of experiments.
        """
        self.client = LLMmanager(api_key, model)
        self.api_key = api_key
        self.model = model
        self.f = f  # evaluation function, provides a string as feedback, a numerical value (higher is better), and a possible error string.
        self.role_prompt = role_prompt
        if role_prompt == "":
            self.role_prompt = "You are a highly skilled computer scientist in the field of natural computing. Your task is to design novel metaheuristic algorithms to solve black box optimization problems."
        if task_prompt == "":
            self.task_prompt = """
The optimization algorithm should handle a wide range of tasks, which is evaluated on the BBOB test suite of 24 noiseless functions. Your task is to write the optimization algorithm in Python code. The code should contain an `__init__(self, budget, dim)` function and the function `def __call__(self, func)`, which should optimize the black box function `func` using `self.budget` function evaluations.
The func() can only be called as many times as the budget allows, not more. Each of the optimization functions has a search space between -5.0 (lower bound) and 5.0 (upper bound). The dimensionality can be varied.
An example of such code (a simple random search), is as follows:
```
import numpy as np

class RandomSearch:
    def __init__(self, budget=10000, dim=10):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        self.f_opt = np.Inf
        self.x_opt = None
        for i in range(self.budget):
            x = np.random.uniform(func.bounds.lb, func.bounds.ub)
            
            f = func(x)
            if f < self.f_opt:
                self.f_opt = f
                self.x_opt = x
            
        return self.f_opt, self.x_opt
```
Give an excellent and novel heuristic algorithm to solve this task and also give it a name. Give the response in the format:
# Name: <name>
# Code: <code>
"""
        else:
            self.task_prompt = task_prompt
        self.feedback_prompt = feedback_prompt
        if feedback_prompt == "":
            self.feedback_prompt = (
                f"Either refine or redesign to improve the solution (and give it a distinct name). Give the response in the format:\n"
                f"# Name: <name>\n"
                f"# Code: <code>"
            )
        self.budget = budget
        self.elitism = elitism
        self.generation = 0
        self.best_solution = None
        self.best_fitness = -np.Inf
        self.best_error = ""
        self.last_error = ""
        self.last_solution = ""
        self.history = ""
        self.log = log
        self.HPO = HPO
        if self.log:
            modelname = self.model.replace(":", "_")
            self.logger = ExperimentLogger(f"{modelname}-ES {experiment_name}")
        else:
            self.logger = None

    def populate_individual(self, session_messages):
        """
        Populates an individual with the given prompt and evaluates its fitness.
        """
        try:
            solution, name, algorithm_name_long, config_space = self.llm(
                session_messages
            )
            self.last_solution = solution
            (
                self.last_feedback,
                self.last_fitness,
                self.last_error,
            ) = self.evaluate_fitness(solution, name, algorithm_name_long, config_space)

        except NoCodeException:
            self.last_fitness = -np.Inf
            self.last_feedback = "No code was extracted."
        except Exception as e:
            self.last_fitness = -np.Inf
            self.last_error = repr(e) + traceback.format_exc()
            self.last_feedback = f"An exception occured: {self.last_error}."
            print(self.last_error)

    def initialize(self):
        """
        Initializes the evolutionary process by generating the first parent program.
        """
        self.last_error = ""
        session_messages = [
            {"role": "system", "content": self.role_prompt},
            {"role": "user", "content": self.task_prompt},
        ]
        self.populate_individual(session_messages)

        self.generation += 1
        self.best_solution = self.last_solution
        self.best_fitness = self.last_fitness
        self.best_error = self.last_error
        self.best_feedback = self.last_feedback

    def llm(self, session_messages):
        """
        Interacts with a language model to generate or mutate solutions based on the provided session messages.

        Args:
            session_messages (list): A list of dictionaries with keys 'role' and 'content' to simulate a conversation with the language model.

        Returns:
            tuple: A tuple containing the new algorithm code, its class name, its full descriptive name and an optional configuration space object.

        Raises:
            NoCodeException: If the language model fails to return any code.
            Exception: Captures and logs any other exceptions that occur during the interaction.
        """
        if self.log:
            self.logger.log_conversation(
                "LLaMEA", "\n".join([d["content"] for d in session_messages])
            )

        message = self.client.chat(session_messages)

        if self.log:
            self.logger.log_conversation(self.model, message)
        new_algorithm = self.extract_algorithm_code(message)

        config_space = None
        if self.HPO:
            config_space = self.extract_configspace(message)

        algorithm_name = re.findall(
            "class\\s*(\\w*)(?:\\(\\w*\\))?\\:", new_algorithm, re.IGNORECASE
        )[0]
        algorithm_name_long = self.extract_algorithm_name(message)
        if algorithm_name_long == "":
            algorithm_name_long = algorithm_name
        # todo rename algorithm
        self.last_solution = message
        # extract algorithm name and algorithm
        return new_algorithm, algorithm_name, algorithm_name_long, config_space

    def evaluate_fitness(self, solution, name, long_name, config_space=None):
        """
        Evaluates the fitness of the provided solution by invoking the evaluation function `f`.
        This method handles error reporting and logs the feedback, fitness, and errors encountered.

        Args:
            solution (str): The solution code to evaluate.
            name (str): The name of the algorithm.
            long_name (str): The full descriptive name of the algorithm.
            config_space (ConfigSpace): An optional configuration space object to perform HPO on the algorithm.

        Returns:
            tuple: A tuple containing feedback (string), fitness (float), and error message (string).
        """
        # Implement fitness evaluation and error handling logic.
        if self.log:
            self.logger.log_code(self.generation, name, solution)
        complete_log = {}
        if self.HPO:
            if self.log:
                self.logger.log_configspace(self.generation, name, config_space)
            feedback, fitness, error, complete_log = self.f(
                solution, name, long_name, config_space, self.logger
            )
        else:
            feedback, fitness, error, complete_log = self.f(
                solution, name, long_name, self.logger
            )
        if self.log:
            complete_log["_generation"] = self.generation
            complete_log["_name"] = name
            complete_log["_fitness"] = fitness
            complete_log["_error"] = error
            complete_log["_feedback"] = feedback
            complete_log["_solution"] = solution
            complete_log["_long_name"] = long_name
            self.logger.log_others(complete_log)
        self.history += f"\nYou already tried {long_name}, with score: {fitness}"
        if error != "":
            self.history += f" with error: {error}"
        return feedback, fitness, error

    def construct_prompt(self):
        """
        Constructs a new session prompt for the language model based on the best or the latest solution,
        depending on whether elitism is enabled.

        Returns:
            list: A list of dictionaries simulating a conversation with the language model for the next evolutionary step.
        """
        if self.elitism:
            solution = f"The best so far algorithm is as follows: \n```\n{self.best_solution}\n```\n"
            feedback = self.best_feedback
        else:
            solution = f"The last tried algorithm is as follows: \n```\n{self.last_solution}\n```\n"
            feedback = self.last_feedback

        session_messages = [
            {"role": "system", "content": self.role_prompt},
            {"role": "user", "content": self.task_prompt},
            {"role": "user", "content": self.history},
            {"role": "assistant", "content": solution},
            {"role": "user", "content": feedback},
            {"role": "user", "content": self.feedback_prompt},
        ]
        # Logic to construct the new prompt based on current evolutionary state.
        return session_messages

    def update_best(self):
        """
        Updates the record of the best solution found so far if the latest solution has a higher fitness.
        This method checks and compares the fitness of the latest solution against the best-known fitness.
        """
        if self.best_fitness <= self.last_fitness or self.last_fitness == -np.Inf:
            self.best_solution = self.last_solution
            self.best_fitness = self.last_fitness
            self.best_error = self.last_error

    def extract_configspace(self, message):
        """
        Extracts the configuration space definition in json from a given message string using regular expressions.

        Args:
            message (str): The message string containing the algorithm code.

        Returns:
            ConfigSpace: Extracted configuration space object.
        """
        print("Extracting configuration space")
        pattern = r"```\n*json\n(.*?)\n```"
        c = None
        for m in re.finditer(pattern, message, re.DOTALL | re.IGNORECASE):
            print("group", m.group(1))
            try:
                c = ConfigurationSpace(eval(m.group(1)))
            except Exception as e:
                print(e.with_traceback)
                pass
        return c

    def extract_algorithm_code(self, message):
        """
        Extracts algorithm code from a given message string using regular expressions.

        Args:
            message (str): The message string containing the algorithm code.

        Returns:
            str: Extracted algorithm code.

        Raises:
            NoCodeException: If no code block is found within the message.
        """
        pattern = r"```(?:python)?\n(.*?)\n```"
        match = re.search(pattern, message, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1)
        else:
            print(message, "contained no ``` code block")
            raise NoCodeException

    def extract_algorithm_name(self, message):
        """
        Extracts algorithm name from a given message string using regular expressions.

        Args:
            message (str): The message string containing the algorithm name and code.

        Returns:
            str: Extracted algorithm name or empty string.
        """
        pattern = r"#\s*Name:\s*(.*)"
        match = re.search(pattern, message, re.IGNORECASE)
        if match:
            return match.group(1)
        else:
            return ""

    def run(self):
        """
        Main loop to evolve the solutions until the evolutionary budget is exhausted.
        The method iteratively refines solutions through interaction with the language model,
        evaluates their fitness, and updates the best solution found.

        Returns:
            tuple: A tuple containing the best solution and its fitness at the end of the evolutionary process.
        """
        self.initialize()
        while self.generation < self.budget:
            new_prompt = self.construct_prompt()
            self.populate_individual(new_prompt)
            self.update_best()
            self.generation = self.generation + 1

        return self.best_solution, self.best_fitness
