"""Shared pytest fixtures for the API tests."""

import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture
def client():
    """Provide a TestClient with the app's lifespan running.

    Using TestClient as a context manager triggers startup, so
    app.state.features and app.state.metadata are loaded before each test.
    Without the `with` block, those would be missing and /recommend and
    /search would fail with an AttributeError.
    """
    with TestClient(app) as test_client:
        yield test_client
