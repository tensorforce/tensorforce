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

Some environments require additional packages, for which there are also options available (`mazeexp`, `gym`, `retro`, `vizdoom`; or `envs` for all environments), however, some require other tools to be installed separately (see [environments documentation](../environments/environment.html)).
