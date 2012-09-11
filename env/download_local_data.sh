# download locally cached data in "_local" from remote server

LOCAL_FNAME="$TG_BASE_DIR/tg_local_data.tar.bz2"
REMOTE_FNAME="translate.idi.ntnu.no:/export/a/emarsi/tg-data/tg_local_data.tar.bz2"
scp "$REMOTE_FNAME" "$LOCAL_FNAME" 
tar xvyf "$LOCAL_FNAME" -C "$TG_BASE_DIR" 
rm -v "$LOCAL_FNAME"
    