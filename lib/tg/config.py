"""
configuration

Reads default config from $TG_BASE_DIR/env/tg-default.cfg (under version control) and
overrides this with user settings from TG_BASE_DIR/env/_tg.cfg (not under version control). 
"""

import logging
from os import getenv
from os.path import join

from configobj import ConfigObj

log = logging.getLogger(__name__)

config = ConfigObj()

tg_base_dir = getenv("TG_BASE_DIR")

if not tg_base_dir:
    raise NameError("environment variable TG_BASE_DIR is undefined")

log.debug("TG_BASE_DIR = " + tg_base_dir)

# inject env var TG_BASE_DIR in config 
config["TG_BASE_DIR"] = tg_base_dir
default_cfg_fname = join(tg_base_dir, "env", "tg-default.cfg")

log.debug("reading default config from " + default_cfg_fname)    
# Temporarily switch off interpolation, because while reading the default
# config, TG_BASE_DIR is not yet defined. 
config.merge(ConfigObj(default_cfg_fname, file_error=True,
                       interpolation=False))
config.interpolation = True

# overide defaults with local user settings
user_cfg_fname = join(tg_base_dir, "env", "_tg.cfg")

try:    
    # Temporarily switch off interpolation, because while reading the user
    # config, variables from the default config are not yet defined.
    config.merge(ConfigObj(user_cfg_fname, file_error=True,
                           interpolation=False))
except IOError:
    pass
else:
    log.debug("applied user config from " + user_cfg_fname)
    config.interpolation = True
    
if log.isEnabledFor(logging.DEBUG):
    from StringIO import StringIO
    str_buf = StringIO()
    config.write(str_buf)
    log.debug("configuration:\n" + str_buf.getvalue())
    str_buf.close()
