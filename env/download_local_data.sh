# download locally cached data in "_local" from remote server

LOCAL_FNAME="$TG_BASE_DIR/tg_local_data.tar.bz2"
REMOTE_FNAME="translate.idi.ntnu.no:/export/a/emarsi/tg-data/tg_local_data.tar.bz2"

if [[ $OSTYPE == darwin* ]]; then
    TAR_OPTS=xvyf
else
    # gnu-linux
    TAR_OPTS=xvjf
fi

scp "$REMOTE_FNAME" "$LOCAL_FNAME" 
tar $TAR_OPTS "$LOCAL_FNAME" -C "$TG_BASE_DIR" 
rm -v "$LOCAL_FNAME"
    