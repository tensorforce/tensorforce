import unittest

from ..config import Configuration
from tensorforce import TensorForceError


class TestConfiguration(unittest.TestCase):

    def test_defaults_allowed(self):
        config = Configuration(allow_defaults=True, a=1, b=2)
        config.default({'c': 3})
        self.assertEqual(config.c, 3)

    def test_no_defaults_raises(self):
        config = Configuration(allow_defaults=False, a=1, b=2)
        with self.assertRaises(TensorForceError):
            config.default({'c': 3})

    def test_defaults_allowed_with_json_load_relying_upon_param_default(self):
        config = Configuration.from_json('config_test.json')
        config.default({'c': 3})
        self.assertEqual(config.c, 3)

    def test_defaults_allowed_with_json_load_specifying_param_default(self):
        config = Configuration.from_json('config_test.json', allow_defaults=True)
        config.default({'c': 3})
        self.assertEqual(config.c, 3)

    def test_defaults_disallowed_with_json_load_specifying_param_default(self):
        config = Configuration.from_json('config_test.json', allow_defaults=False)
        with self.assertRaises(TensorForceError):
            config.default({'c': 3})

    def test_default_to_provided_param_is_ok(self):
        config = Configuration.from_json('config_test.json', allow_defaults=False)
        config.default({'a': 'boo!'})
        self.assertEqual(config.a, 1)

    def test_conflicting_desires_raises(self):
        with self.assertRaises(TensorForceError):
            config = Configuration.from_json('config_test_with_flag.json', allow_defaults=False)


if __name__ == "__main__":
    unittest.main()
