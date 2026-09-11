"""Test logging configuration."""

import logging
from agent.logging_config import setup_logging, get_logger


def test_logging_setup():
    """Test that logging can be configured without errors."""
    # This should not raise an exception
    setup_logging()

    # Test that we can get a logger
    logger = get_logger("test")
    assert isinstance(logger, logging.Logger)
    assert logger.name == "agent.test"


def test_logger_levels():
    """Test that logger levels are set correctly."""
    setup_logging()
    logger = get_logger("test")

    # Child loggers start with level NOTSET (0) and inherit effective level from parent
    # Check that the effective level matches the parent's level
    assert logger.getEffectiveLevel() == logging.getLogger("agent").getEffectiveLevel()

    # Also verify the parent logger is set to the expected level from settings
    agent_logger = logging.getLogger("agent")
    expected_level = getattr(logging, "INFO")  # Default from settings
    assert agent_logger.getEffectiveLevel() == expected_level