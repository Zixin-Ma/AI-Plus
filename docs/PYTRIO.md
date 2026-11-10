# PyTRIO environment

All commands below run on BJTU in `/opt/data/private/mzx/mzx/AI +`.
The SDK runs the control loop on BJTU and connects to the PyTRIO service.

## Python environment

The project uses the named Conda environment `ai-plus` with Python 3.12.12.
If the current terminal shows `((.venv) )`, run `deactivate` first.

```bash
cd "/opt/data/private/mzx/mzx/AI +"
source /root/miniconda3/etc/profile.d/conda.sh
conda activate ai-plus
python -m pip check
python -c 'from importlib.metadata import version; print(version("pytrio"))'
trio --help
```

To recreate the environment on BJTU:

```bash
cd "/opt/data/private/mzx/mzx/AI +"
source /root/miniconda3/etc/profile.d/conda.sh
conda create --name ai-plus --override-channels -c https://repo.anaconda.com/pkgs/main python=3.12.12 pip
conda activate ai-plus
python -m pip install --no-cache-dir -r requirements-pytrio.txt
```

Respect the pinned SDK version. Install optional training libraries only as a
specific experiment requires them. Keep downloaded models, datasets, logs, and
outputs under the repository's existing ignored directories.

## Authentication

In an interactive BJTU terminal, activate the environment and run:

```bash
umask 077
trio login
```

Paste the full API Key copied from the PyTRIO console when prompted. Input is
hidden. A masked key from a screenshot cannot authenticate. The command validates
the key with the service before persisting it. Do not pass the key as a command
argument, put it in source code, or commit it to Git.

The current SSH account uses `/root/.pytrio/config.toml` for the default profile;
this is outside the project repository. Keep the directory private (`0700`) and
any credential file private (`0600`).

Inspect local profile state, then verify the saved credentials with the service:

```bash
trio env show
trio login
```

`trio env show` reports local configuration; a successful `trio login` is the
credential validation. An optional read-only API check after login is:

```bash
python -c 'import pytrio as trio; print(trio.ServiceClient().get_supported_models())'
```

Authentication and model discovery do not launch training or inference. Run an
actual workload only when its model, data, and experiment settings are selected.

## Agent skill

Official PyTRIO skill 0.2.3 is installed on BJTU at
`/root/.agents/skills/pytrio-skill`, with the Codex discovery link
`/root/.codex/skills/pytrio-skill`. Read its `SKILL.md`, then the relevant references
and examples. The complete skill bundle and Python examples remain on the server.

The Windows Codex entry `pytrio-bjtu` contains only lightweight routing rules and
loads the official skill over SSH. It does not contain project code or SDK files.
The local entry is available on the next Codex turn; refresh the skill list or
start a new chat if the current interface has not picked it up.

Official sources:

- https://docs.pytrio.com/docs/quick-start
- https://pypi.org/project/pytrio/0.2.9/
- https://github.com/SwanHubX/pytrio-skill

## Storage and synchronization

The `ai-plus` Conda environment, credentials, and globally installed skill
live outside the repository. The legacy `.venv/` and `tmp/` are ignored by Git.
This guide and the dependency specification use the existing hourly Git sync.
