# Setup env for transglobal

# Default value; set other TG_BASE_DIR is your .bash or .profile
: ${TG_BASE_DIR:="/Users/erwin/Projects/Transglobal/github/transglobal"}
export TG_BASE_DIR

# append lib dir to Python path 
LIB_DIR="$TG_BASE_DIR/lib"
export PYTHONPATH="$LIB_DIR:$PYTHONPATH"


    

