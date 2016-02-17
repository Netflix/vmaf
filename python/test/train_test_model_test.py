__copyright__ = "Copyright 2016, Netflix, Inc."
__license__ = "LGPL Version 3"

import unittest


@unittest.skip("zzzzzzzzzzzzz")
class TrainTestModelTest(unittest.TestCase):
    def test_something(self):
        self.assertEqual(True, False)


if __name__ == '__main__':
    unittest.main()
