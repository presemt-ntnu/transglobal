# Activate shell and Anaconda python environment for transglobal

# Usage: source activate_anaconda.sh

SCRIPT_PATH="$(dirname "$BASH_SOURCE[0]")"
TG_BASE_DIR="$(unset CDPATH && cd "$SCRIPT_PATH" && pwd)"

# setup shell environment
source "$TG_BASE_DIR/env/setup_shell_env.sh"

# activate Conda virtual python environment 
source activate transglobal
