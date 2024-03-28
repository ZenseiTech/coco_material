## Setting Poetry

https://python-poetry.org/docs/

python3 -m venv $VENV_PATH
source ./$VENV_PATH/bin/activate
pip install -U pip setuptools
pip install poetry

### Auto-loaded (recommended)

poetry completions bash >> ~/.bash_completion

### Installing dependencies

poetry install
