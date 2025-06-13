# set SFPI release version information

sfpi_version=v6.12.0-insn-synth
sfpi_url=https://github.com/tenstorrent/sfpi/releases/download

# convert md5 file into these variables
# sed 's/^\([0-9a-f]*\) \*sfpi-\([a-z0-9_A-Z]*\)\.\([a-z]*\)$/sfpi_\2_\3_md5=\1/'
sfpi_x86_64_Linux_txz_md5=19d9569c5ed4ef46a290ab5a4deea290
sfpi_x86_64_Linux_deb_md5=8bd9a506cde415744e96f579a4b32a68
sfpi_x86_64_Linux_rpm_md5=4ceebf136e228ea37b3c8f744c393bba
