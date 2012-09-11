# upload locally cached data in "_local" to remote server

LOCAL_FNAME="$TG_BASE_DIR/tg_local_data.tar.bz2"
REMOTE_FNAME="translate.idi.ntnu.no:/export/a/emarsi/tg-data/tg_local_data.tar.bz2"
tar cvyf "$LOCAL_FNAME" -C "$TG_BASE_DIR" --exclude='*.DS_Store' _local
scp "$LOCAL_FNAME" "$REMOTE_FNAME"
rm -v "$LOCAL_FNAME"
    