# upload locally cached data in "_local" to remote server

LOCAL_FNAME="$TG_BASE_DIR/tg_local_data.tar.bz2"
REMOTE_FNAME="translate.idi.ntnu.no:/export/a/emarsi/tg-data/tg_local_data.tar.bz2"

if [[ $OSTYPE == darwin* ]]; then
    TAR_OPTS=cvyf
else
    # gnu-linux
    TAR_OPTS=cvjf
fi

tar $TAR_OPTS  "$LOCAL_FNAME" -C "$TG_BASE_DIR" --exclude='*.DS_Store' --exclude='*~' _local
scp "$LOCAL_FNAME" "$REMOTE_FNAME"
rm -v "$LOCAL_FNAME"
    