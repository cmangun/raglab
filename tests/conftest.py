import tempfile

import pytest

from raglab.build import build_lab
from raglab.core.types import Principal


@pytest.fixture(scope="session")
def lab():
    return build_lab("corpus")


@pytest.fixture(scope="session")
def pg_dsn():
    pgserver = pytest.importorskip("pgserver")
    server = pgserver.get_server(tempfile.mkdtemp())
    yield server.get_uri()
    server.cleanup()


@pytest.fixture(scope="session")
def pg_lab(pg_dsn):
    return build_lab("corpus", dsn=pg_dsn)


@pytest.fixture(params=["memory", "postgres"])
def any_lab(request, lab):
    return lab if request.param == "memory" else request.getfixturevalue("pg_lab")


QUALITY = Principal.of("q", "quality")
COMMERCIAL = Principal.of("c", "commercial")
