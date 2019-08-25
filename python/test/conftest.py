import pytest


def pytest_collection_modifyitems(items):
    """
    Dynamically mark tests based on their file name:
    - *_test.py: main test (always exercise)
    - *_extratest.py: exercised only when testing with ffmpeg
    - *_libtest.py: exercised only to test testlib
    """
    for item in items:
        if "_extratest" in item.nodeid:
            item.add_marker(pytest.mark.extra)

        elif "_libtest" in item.nodeid:
            item.add_marker(pytest.mark.lib)

        else:
            item.add_marker(pytest.mark.main)
