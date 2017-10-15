import unittest

from tensorforce.config import Configuration
from tensorforce import TensorForceError


test_config = """
{
  "a": 1,
  "b": 2
}
"""

test_config_with_flag = """
{
  "a": 1,
  "b": 2,
  "allow_defaults": true
}
"""


class TestConfiguration(unittest.TestCase):
    pass
    #TODO new config tests

    # def test_defaults_allowed(self):
    #     config = Configuration(allow_defaults=True, a=1, b=2)
    #     config.default({'c': 3})
    #     self.assertEqual(config.c, 3)
    #
    # def test_no_defaults_raises(self):
    #     config = Configuration(allow_defaults=False, a=1, b=2)
    #     with self.assertRaises(TensorForceError):
    #         config.default({'c': 3})
    #
    # def test_defaults_allowed_with_json_load_relying_upon_param_default(self):
    #     config = Configuration.from_json_string(test_config)
    #     config.default({'c': 3})
    #     self.assertEqual(config.c, 3)
    #
    # def test_defaults_allowed_with_json_load_specifying_param_default(self):
    #     config = Configuration.from_json_string(test_config, allow_defaults=True)
    #     config.default({'c': 3})
    #     self.assertEqual(config.c, 3)
    #
    # def test_defaults_disallowed_with_json_load_specifying_param_default(self):
    #     config = Configuration.from_json_string(test_config, allow_defaults=False)
    #     with self.assertRaises(TensorForceError):
    #         config.default({'c': 3})
    #
    # def test_default_to_provided_param_is_ok(self):
    #     config = Configuration.from_json_string(test_config, allow_defaults=False)
    #     config.default({'a': 'boo!'})
    #     self.assertEqual(config.a, 1)
    #
    # def test_conflicting_desires_raises(self):
    #     with self.assertRaises(TensorForceError):
    #         Configuration.from_json_string(test_config_with_flag, allow_defaults=False)
