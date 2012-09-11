# setup virtual python environment
ENV=_python
python '../bin/virtualenv.py' --verbose --distribute --system-site-packages --prompt="{transglobal}" "$ENV"

# install required packages with pip
"$ENV/bin/pip" install --requirement='requirements_osx.txt'

