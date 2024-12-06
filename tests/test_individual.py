import pytest
import uuid
import json
from llamea import (
    Individual,
)  # Replace with the actual module where Individual is defined


def test_individual_initialization():
    # Test initialization of an individual
    individual = Individual(
        solution="def my_solution(): pass",
        name="MySolution",
        description="This is a test solution.",
        generation=1,
    )

    assert isinstance(individual.id, str)  # Check if ID is a string
    assert uuid.UUID(individual.id)  # Ensure the ID is a valid UUID
    assert individual.solution == "def my_solution(): pass"
    assert individual.name == "MySolution"
    assert individual.description == "This is a test solution."
    assert individual.generation == 1
    assert individual.fitness is None  # Default should be None
    assert individual.feedback == ""  # Default should be empty string
    assert individual.error == ""  # Default should be empty string


def test_add_metadata():
    # Test adding metadata to an individual
    individual = Individual()
    individual.add_metadata("source", "LLM-generated")
    individual.add_metadata("version", 1.0)

    assert "source" in individual.metadata
    assert "version" in individual.metadata
    assert individual.metadata["source"] == "LLM-generated"
    assert individual.metadata["version"] == 1.0


def test_copy_individual():
    # Test copying an individual
    individual = Individual(
        solution="def my_solution(): pass",
        name="MySolution",
        description="This is a test solution.",
        generation=1,
    )
    individual.add_metadata("key", "value")

    new_individual = individual.copy()

    # Check if new_individual is a copy and has a new unique ID
    assert new_individual.id != individual.id
    assert (
        new_individual.parent_id == individual.id
    )  # The parent_id of the new individual should be the original individual's ID
    assert new_individual.solution == individual.solution
    assert new_individual.name == individual.name
    assert new_individual.description == individual.description
    assert new_individual.metadata == individual.metadata  # Metadata should be copied


def test_to_dict():
    # Test converting an individual to a dictionary
    individual = Individual(
        solution="def my_solution(): pass",
        name="MySolution",
        description="This is a test solution.",
        generation=1,
    )
    individual.add_metadata("source", "LLM-generated")

    individual_dict = individual.to_dict()

    assert isinstance(individual_dict, dict)
    assert individual_dict["solution"] == "def my_solution(): pass"
    assert individual_dict["name"] == "MySolution"
    assert individual_dict["description"] == "This is a test solution."
    assert individual_dict["generation"] == 1
    assert "metadata" in individual_dict
    assert individual_dict["metadata"] == {"source": "LLM-generated"}


def test_to_json():
    # Test converting an individual to JSON
    individual = Individual(
        solution="def my_solution(): pass",
        name="MySolution",
        description="This is a test solution.",
        generation=1,
    )
    individual.add_metadata("source", "LLM-generated")

    individual_json = individual.to_json()

    assert isinstance(individual_json, str)

    # Convert back to dict to validate content
    individual_dict = json.loads(individual_json)

    assert individual_dict["solution"] == "def my_solution(): pass"
    assert individual_dict["name"] == "MySolution"
    assert individual_dict["description"] == "This is a test solution."
    assert individual_dict["generation"] == 1
    assert individual_dict["metadata"] == {"source": "LLM-generated"}


def test_mutation():
    # Test mutation prompt assignment
    individual = Individual()
    mutation_prompt = "Refine the strategy of the solution."
    individual.set_mutation_prompt(mutation_prompt)

    assert individual.mutation_prompt == mutation_prompt


def test_default_values():
    # Test default values of an individual
    individual = Individual()

    assert individual.solution == ""
    assert individual.name == ""
    assert individual.description == ""
    assert individual.configspace is None
    assert individual.fitness is None
    assert individual.feedback == ""
    assert individual.error == ""
    assert individual.parent_id is None
    assert individual.metadata == {}
