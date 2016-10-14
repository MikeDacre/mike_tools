#!/bin/bash

git clone https://github.com/yyuu/pyenv.git ~/.pyenv
git clone https://github.com/yyuu/pyenv-virtualenv.git ~/.pyenv/plugins/pyenv-virtualenv

export PYENV_ROOT="$HOME/.pyenv"
export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init -)"
eval "$(pyenv virtualenv-init -)"

~/.pyenv/bin/pyenv install anaconda3-4.1.1
~/.pyenv/bin/pyenv shell anaconda3-4.1.1
conda update conda ipython pandas scipy numpy sqlalchemy

git clone https://github.com/jawshooah/pyenv-default-packages.git ~/.pyenv/plugins/pyenv-default-packages

cat > ~/.pyenv/default-packages << EOF
argcomplete >= 1.0.0
astroid >= 1.4.8
cycler >= 0.10.0
decorator >= 4.0.10
dictionaries >= 0.0.1
dill >= 0.2.5
ipython >= 5.1.0
ipyparallel >= 5.2.0
ipython-genutils >= 0.1.0
isort >= 4.2.5
lazy-object-proxy >= 1.2.2
matplotlib >= 1.5.3
mccabe >= 0.5.2
numpy >= 1.11.2
pandas >= 0.19.0
pexpect >= 4.2.1
pickleshare >= 0.7.4
pip >= 8.1.2
plotly >= 1.12.9
ptyprocess >= 0.5.1
Pygments >= 2.1.3
pylint >= 1.6.4
pyparsing >= 2.1.10
python-dateutil >= 2.5.3
pytz >= 2016.7
requests >= 2.11.1
scipy >= 0.18.1
setuptools >= 28.3.0
simplegeneric >= 0.8.1
six >= 1.10.0
SQLAlchemy >= 1.1.1
tqdm >= 4.8.4
wcwidth >= 0.1.7
wrapt >= 1.10.8
xlrd >= 1.0.0
xlwt >= 1.1.2
XlsxWriter >= 0.9.2
EOF

pip install -r ~/.pyenv/default-packages
~/.pyenv/bin/pyenv shell --unset

env PYTHON_CONFIGURE_OPTS="--enable-shared" ~/.pyenv/bin/pyenv install 3.5.2
env PYTHON_CONFIGURE_OPTS="--enable-shared" ~/.pyenv/bin/pyenv install 2.7.12

git clone https://github.com/doloopwhile/pyenv-register.git ~/.pyenv/plugins/pyenv-register
~/.pyenv/bin/pyenv register `which python3`
~/.pyenv/bin/pyenv register `which python2`

~/.pyenv/bin/pyenv global 2.5.2 2.7.12

git clone https://github.com/yyuu/pyenv-which-ext.git ~/.pyenv/plugins/pyenv-which-ext
git clone https://github.com/s1341/pyenv-alias.git ~/.pyenv/plugins/pyenv-alias
git clone git://github.com/yyuu/pyenv-update.git ~/.pyenv/plugins/pyenv-update

mv ~/.pyenv/default-packages ~/.pyenv/default-packages-off
