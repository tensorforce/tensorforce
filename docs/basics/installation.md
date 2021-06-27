Installation
============


A stable version of Tensorforce is periodically updated on PyPI and installed as follows:

```bash
pip3 install tensorforce
```

To always use the latest version of Tensorforce, install the GitHub version instead:

```bash
git clone https://github.com/tensorforce/tensorforce.git
cd tensorforce
pip3 install -e .
```

Environments require additional packages for which there are setup options available (`ale`, `gym`, `retro`, `vizdoom`, `carla`; or `envs` for all environments), however, some require additional tools to be installed separately (see [environments documentation](http://tensorforce.readthedocs.io)). Other setup options include `tfa` for [TensorFlow Addons](https://www.tensorflow.org/addons) and `tune` for [HpBandSter](https://github.com/automl/HpBandSter) required for the `tune.py` script.


**Note on GPU usage:** Different from (un)supervised deep learning, RL does not always benefit from running on a GPU, depending on environment and agent configuration. In particular for RL-typical environments with low-dimensional state spaces (i.e., no images), one usually gets better performance by running on CPU only. Consequently, Tensorforce is configured to run on CPU by default, which can be changed via the agent's `config` argument, for instance, `config=dict(device='GPU')`.


**M1 Macs**

At the moment Tensorflow cannot be installed on M1 Macs directly. You need to follow [Apple's guide](https://developer.apple.com/metal/tensorflow-plugin/) to install `tensorflow-macos` instead.

Then, since Tensorforce has `tensorflow` as its dependency and not `tensorflow-macos`, you need to install all Tensorforce's dependencies from [requirements.txt](https://github.com/tensorforce/tensorforce/blob/master/requirements.txt) manually (except for `tensorflow == 2.5.0` of course).

In the end, install tensorforce while forcing pip to ignore its dependencies:
```
pip3 install tensorforce==0.6.4 --no-deps
```


**Dockerfile**

If you want to use Tensorforce within a Docker container, the following is a minimal `Dockerfile` to get started:

```
FROM python:3.8
RUN \
  pip3 install tensorforce
```

Or alternatively for the latest version:

```
FROM python:3.8
RUN \
  git clone https://github.com/tensorforce/tensorforce.git && \
  pip3 install -e tensorforce
```

Subsequently, the container can be built via:

```bash
docker build .
```
