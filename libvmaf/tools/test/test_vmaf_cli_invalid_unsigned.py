# Copyright 2026 Mangesh Raut
#
# Licensed under the BSD+Patent License (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://opensource.org/licenses/BSDplusPatent
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import subprocess
import sys
import unittest


class InvalidUnsignedTest(unittest.TestCase):
    def test_invalid_long_options_exit_with_their_argument_error(self):
        for option in ("--threads", "--subsample", "--cpumask"):
            for value in ("abc", "1x", ""):
                for args in ([option, value], [option + "=" + value]):
                    with self.subTest(args=args):
                        result = subprocess.run(
                            [VMAF, *args],
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE,
                            universal_newlines=True,
                            timeout=10,
                        )
                        # A crash is not the normal usage-error exit status.
                        self.assertEqual(result.returncode, 1, result.stderr)
                        self.assertIn(
                            'Invalid argument "{}" for option {}; should be an integer'.format(
                                value, option
                            ),
                            result.stderr,
                        )


if __name__ == "__main__":
    VMAF = sys.argv.pop(1)
    unittest.main()
