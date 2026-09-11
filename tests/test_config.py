"""Test configuration loading."""

import os
from agent.config import Settings


def test_settings_defaults():
    """Test that settings load with default values."""
    # Clear relevant environment variables to test defaults
    for key in list(os.environ.keys()):
        if key.startswith("APP_") or key in ["DEBUG", "LOG_LEVEL", "MAX_CONCURRENT_TASKS", "TASK_TIMEOUT_SECONDS"]:
            del os.environ[key]

    settings = Settings()
    assert settings.app_name == "Personal AI Agent"
    assert settings.app_version == "0.1.0"
    assert settings.debug is False
    assert settings.log_level == "INFO"
    assert settings.max_concurrent_tasks == 10
    assert settings.task_timeout_seconds == 300


def test_settings_from_env():
    """Test that settings can be overridden by environment variables."""
    os.environ["APP_NAME"] = "Test Agent"
    os.environ["APP_VERSION"] = "1.0.0"
    os.environ["DEBUG"] = "true"
    os.environ["LOG_LEVEL"] = "DEBUG"
    os.environ["MAX_CONCURRENT_TASKS"] = "5"
    os.environ["TASK_TIMEOUT_SECONDS"] = "100"

    try:
        settings = Settings()
        assert settings.app_name == "Test Agent"
        assert settings.app_version == "1.0.0"
        assert settings.debug is True
        assert settings.log_level == "DEBUG"
        assert settings.max_concurrent_tasks == 5
        assert settings.task_timeout_seconds == 100
    finally:
        # Clean up environment variables
        for key in ["APP_NAME", "APP_VERSION", "DEBUG", "LOG_LEVEL", "MAX_CONCURRENT_TASKS", "TASK_TIMEOUT_SECONDS"]:
            if key in os.environ:
                del os.environ[key]