#!/usr/bin/env bash
set -e

[ -e dist ] || mkdir dist

# DATASET_URL=''
DATASET_URL=''

pushd dist
# [ -e fra-eng.zip ] || curl -L -O http://www.manythings.org/anki/fra-eng.zip
# [ -e 'fra.txt' ]   || unzip -o fra-eng.zip
[ -e dataset.txt.gz ] || curl -L -O https://github.com/huan/python-concise-chitchat/releases/download/v0.0.1/dataset.txt.gz
[ -e dataset.txt ] || gzip -d dataset.txt
popd
