#!/bin/sh
# Create api
BIN_DIR=/Users/chiang4/miniconda3/envs/obspy/bin
OUT_DIR=modules
package=../src/tdmtpy

$BIN_DIR/sphinx-apidoc -f -e -P -M -F --templatedir=_templates -o $OUT_DIR $package
