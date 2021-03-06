# setup virtual python environment

# get dir of current script
ENV_DIR="$(unset CDPATH && cd $(dirname "$0") && pwd -P)"
echo $ENV_DIR

PYTHON_ENV="$ENV_DIR/_python"
#python "$ENV_DIR/../bin/virtualenv.py" --verbose --distribute --system-site-packages --prompt="{transglobal}" "$PYTHON_ENV"
virtualenv --verbose --distribute --system-site-packages --prompt="{transglobal}" "$PYTHON_ENV"

# install required packages with pip
"$PYTHON_ENV/bin/pip" install --requirement="$ENV_DIR/requirements.txt"

# Hack: force install of nose in virtual env, even if it already
# exists in system wide packages, because otherwise running the
# "nosetests" commend line script will not run with the Python
# interpreter from the virtual env
"$PYTHON_ENV/bin/pip" install --ignore-installed nose

# clean up
if [ -f distribute*.tar.gz ]
then
    rm -v distribute*.tar.gz
fi
