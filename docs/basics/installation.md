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



**Dockerfile**

If you want to use Tensorforce within a Docker container, the following is a minimal `Dockerfile` to get started:

```
FROM python:3.8
RUN \
  pip install tensorforce
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
