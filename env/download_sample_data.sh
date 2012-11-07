# ad-hoc script to download sample data from remote server

SAMP_DIR="$TG_BASE_DIR/_sample"
REMOTE_BASE="translate.idi.ntnu.no:/export/a/emarsi/corpusmod"
mkdir -p "$SAMP_DIR"

scp "$REMOTE_BASE/en/de/en-de_ambig.tab" "$SAMP_DIR"
scp "$REMOTE_BASE/en/de/en-de_samples.hdf5" "$SAMP_DIR" 

scp "$REMOTE_BASE/de/en/de-en_ambig.tab" "$SAMP_DIR"
scp "$REMOTE_BASE/de/en/de-en_samples.hdf5" "$SAMP_DIR" 

scp "$REMOTE_BASE/no/en/no-en_ambig.tab" "$SAMP_DIR"
scp "$REMOTE_BASE/no/en/no-en_samples.hdf5" "$SAMP_DIR" 

scp "$REMOTE_BASE/no/de/no-de_ambig.tab" "$SAMP_DIR"
scp "$REMOTE_BASE/no/de/no-de_samples.hdf5" "$SAMP_DIR"

scp "$REMOTE_BASE/gr/en/gr-en_ambig.tab" "$SAMP_DIR"
scp "$REMOTE_BASE/gr/en/gr-en_samples.hdf5" "$SAMP_DIR" 

scp "$REMOTE_BASE/gr/de/gr-de_ambig.tab" "$SAMP_DIR"
scp "$REMOTE_BASE/gr/de/gr-de_samples.hdf5" "$SAMP_DIR"

#scp "$REMOTE_BASE/cz/en/cz-en_ambig.tab" "$SAMP_DIR"
#scp "$REMOTE_BASE/cz/en/cz-en_samples.hdf5" "$SAMP_DIR" 

#scp "$REMOTE_BASE/cz/de/cz-de_ambig.tab" "$SAMP_DIR"
#scp "$REMOTE_BASE/cz/de/cz-de_samples.hdf5" "$SAMP_DIR"
