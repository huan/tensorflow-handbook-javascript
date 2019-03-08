#!/usr/bin/env bash
set -e

[ -e dist ] || mkdir dist

pushd dist
[ -e fra-eng.zip ] || curl -L -O http://www.manythings.org/anki/fra-eng.zip
[ -e 'fra.txt' ]   || unzip -o fra-eng.zip
popd
