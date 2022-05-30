import pytest


def pytest_collection_modifyitems(items):
    """
    Dynamically mark tests based on their file name:
    - *_test.py: main test (always exercise)
    - *_extratest.py: exercised only when testing with ffmpeg
    - *_libtest.py: exercised only to test testlib
    """
    for item in items:
        item.add_marker(pytest.mark.main)
