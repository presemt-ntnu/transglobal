# download private data in "_data" from remote server

LOCAL_FNAME="$TG_BASE_DIR/tg_private_data.tar.bz2"
REMOTE_FNAME="translate.idi.ntnu.no:/export/a/emarsi/tg-data/tg_private_data.tar.bz2"
scp "$REMOTE_FNAME" "$LOCAL_FNAME" 
tar xvyf "$LOCAL_FNAME" -C "$TG_BASE_DIR" 
rm -v "$LOCAL_FNAME"
    