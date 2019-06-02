# Notes on python2 / python3 support

A `tox.ini` was added to help provide support for python3 in vmaf.

First draft allows to do this:

- python 3:
    - `tox -e venv`, to get a python3 venv in `./.venv`
    - `tox -e py37` to exercise all tests with python 3.7
- python 2:
    - `tox -e venv2`, to get a python2 venv in `./.venv2`
    - `tox -e py27` to exercise all tests with python 2.7
- you can also run `tox` to exercise all tests with both python 2.7 and 3.7
