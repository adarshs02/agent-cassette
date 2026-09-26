import pytest
from retrace.pipeline.baseline import load_baseline
from retrace.pipeline.workspace import prepare


@pytest.fixture(scope="session")
def baseline():
    return load_baseline()


@pytest.fixture(scope="session")
def healthy_ws(tmp_path_factory):
    return prepare(tmp_path_factory.mktemp("healthy"), fault=None)
