import os
import importlib
import unittest
import argparse


def run_tests(verbosity=1):
    package_path = os.path.dirname(importlib.util.find_spec("nngpr").origin)
    suite = unittest.TestLoader().discover(package_path, pattern="test_*.py")
    res = unittest.TextTestRunner(verbosity=verbosity).run(suite)
    assert res.wasSuccessful()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '--verbosity', default=1, type=int, help="Unittest verbosity level")
    args = parser.parse_args()
    run_tests(verbosity=args.verbosity)
