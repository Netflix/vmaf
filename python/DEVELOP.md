# Notes on python2 / python3 support

A `tox.ini` was added to help provide support for python3. Before running tests using tox, be sure to first run `make` under the root directory.

First draft allows to do this:

- python 3:
    - `tox -e venv`, to get a python3 venv in `./.venv`
    - `tox -e py37` to exercise all tests with python 3.7
- python 2:
    - `tox -e venv2`, to get a python2 venv in `./.venv2`
    - `tox -e py27` to exercise all tests with python 2.7
- you can also run `tox` to exercise all tests with both python 2.7 and 3.7


# Support for python3 status

Search for `TODO python3` to spot which places in code still need attention regarding python3 support.

All tests pass with both python2 and python3, however, a few test cases are disabled currently for python3, spot then with:
 
```bash
grep -r 'reason="TODO python3' .
```

Here's the current list:

- `scipy` v1.3.0 removed some deprecated functions tha vmaf still uses -> need to adapt those before we can upgrade to latest scipy
- tests that rely on `random` do not yield the same results in python2 and python3
- vmaf uses `pickle` to serialize some objects, however "pickle is fickle", and a few objects fail to deserialize in python3
- `YuvReader` needs to be reviewed, it doesn't work in python3
- `map()` and `filter()` yield a generator (instead of list) in python3 -> this implies an extra call `to_list()` throughout the code
  (search for `to_list` to find all the spots where this is done).
  All usages of `map()`/`filter()` should be reviewed to leverage the speed generators offer


# Test coverage

If you run `tox`, a summary test coverage report will be shown in terminal, 
you can also see a full HTML report by looking at `.tox/coverage/index.html`.

This report can be useful to spot any key part of the code that was not exercised (and thus possibly likely to fail under python3)
