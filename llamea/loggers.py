import os
from datetime import datetime

import jsonlines
import numpy as np


class ExperimentLogger:
    def __init__(self, name=""):
        """
        Initializes an instance of the ExperimentLogger.
        Sets up a new logging directory named with the current date and time.

        Args:
            name (str): The name of the experiment.
        """
        self.dirname = self.create_log_dir(name)
        self.attempt = 0

    def create_log_dir(self, name=""):
        """
        Creates a new directory for logging experiments based on the current date and time.
        Also creates subdirectories for IOH experimenter data and code files.

        Returns:
            str: The name of the created directory.
        """
        today = datetime.today().strftime("%m-%d_%H%M%S")
        dirname = f"exp-{today}-{name}"
        os.mkdir(dirname)
        os.mkdir(f"{dirname}/code")
        return dirname

    def log_conversation(self, role, content):
        """
        Logs the given conversation content into a conversation log file.

        Args:
            role (str): Who (the llm or user) said the content.
            content (str): The conversation content to be logged.
        """
        conversation_object = {
            "role": role,
            "time": f"{datetime.now()}",
            "content": content,
        }
        with jsonlines.open(f"{self.dirname}/conversationlog.jsonl", "a") as file:
            file.write(conversation_object)

    def set_attempt(self, attempt):
        self.attempt = attempt

    def log_population(self, population):
        for p in population:
            self.log_code(self.attempt, p["name"], p["solution"])
            self.log_aucs(self.attempt, [p["fitness"]])
            self.attempt += 1

    def log_code(self, attempt, algorithm_name, code):
        """
        Logs the provided code into a file, uniquely named based on the attempt number and algorithm name.

        Args:
            attempt (int): The attempt number of the code execution.
            algorithm_name (str): The name of the algorithm used.
            code (str): The source code to be logged.
        """
        with open(
            f"{self.dirname}/code/try-{attempt}-{algorithm_name}.py", "w"
        ) as file:
            file.write(code)
        self.attempt = attempt

    def log_failed_code(self, attempt, algorithm_name, code):
        """
        Logs the provided code into a file, uniquely named based on the attempt number and algorithm name.

        Args:
            attempt (int): The attempt number of the code execution.
            algorithm_name (str): The name of the algorithm used.
            code (str): The source code to be logged.
        """
        with open(
            f"{self.dirname}/code/fail-{attempt}-{algorithm_name}.py", "w"
        ) as file:
            file.write(code)

    def log_aucs(self, attempt, aucs):
        """
        Logs the given AOCCs (Area Over the Convergence Curve, named here auc) into a file, named based on the attempt number.

        Args:
            attempt (int): The attempt number corresponding to the AOCCs.
            aucs (array_like): An array of AUC scores to be saved.
        """
        np.savetxt(f"{self.dirname}/try-{attempt}-aucs.txt", aucs)
