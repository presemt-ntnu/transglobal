# setup virtual python environment

# get dir of current script
ENV_DIR="$(unset CDPATH && cd $(dirname "$0") && pwd -P)"
echo $ENV_DIR

PYTHON_ENV="$ENV_DIR/_python"
python "$ENV_DIR/../bin/virtualenv.py" --verbose --distribute --system-site-packages --prompt="{transglobal}" "$PYTHON_ENV"

# install required packages with pip
"$PYTHON_ENV/bin/pip" install --requirement="$ENV_DIR/requirements.txt"

# clean up
if [ -f distribute*.tar.gz ]
then
    rm -v distribute*.tar.gz
fi
