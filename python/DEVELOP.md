# Development

## Testing

A `tox.ini` was added to provide unit tests. Before running tests using tox, be sure to first run `make` under the root directory.

Run:

```
tox
```

to run all tests.

## Test coverage

If you run `tox`, a summary test coverage report will be shown in terminal, 
you can also see a full HTML report by looking at `.tox/coverage/index.html`.

This report can be useful to spot any key part of the code that was not exercised (and thus possibly likely to fail under python3)
