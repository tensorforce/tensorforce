Installation
============


A stable version of Tensorforce is periodically updated on PyPI and installed as follows:

```bash
pip install tensorforce
```

To always use the latest version of Tensorforce, install the GitHub version instead:

```bash
git clone https://github.com/tensorforce/tensorforce.git
cd tensorforce
pip install -e .
```

Tensorforce is built on top of [Google's TensorFlow](https://www.tensorflow.org/) and requires that either `tensorflow` or `tensorflow-gpu` is installed, currently as version `1.13.1`. To include the correct version of TensorFlow with the installation of Tensorforce, simply add the flag `tf` for the normal CPU version or `tf_gpu` for the GPU version:

```bash
# PyPI version plus TensorFlow CPU version
pip install tensorforce[tf]

# GitHub version plus TensorFlow GPU version
pip install -e .[tf_gpu]
```

Some environments require additional packages, for which there are also options available (`mazeexp`, `gym`, `retro`, `vizdoom`; or `envs` for all environments), however, some require other tools to be installed (see [environments documentation](../environments/environment.html)).
