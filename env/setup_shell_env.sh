# Setup shell environment for transglobal

# Usage: "source setup_env.sh"

# 1. set TG_BASE_DIR
# get dir of current script
SCRIPT_PATH="$(dirname "$BASH_SOURCE[0]")"
# get base directory
# NB: unset CDPATH, otherwise "cd" echos current directory, which messes things up!  
TG_BASE_DIR="$(unset CDPATH && cd "$SCRIPT_PATH/.." && pwd)"
export TG_BASE_DIR

# 2. append bin dir to system path
BIN_DIR="$TG_BASE_DIR/bin"
export PATH="$BIN_DIR:$PATH"

# 3. append lib dir to Python path 
LIB_DIR="$TG_BASE_DIR/lib"
export PYTHONPATH="$LIB_DIR:$PYTHONPATH"

# 4. fix problem with runnning ipython within virtualenv
# alias ipython="python -c 'import IPython; IPython.embed()'"
# Deprecated? Does not work iPython Notebook.