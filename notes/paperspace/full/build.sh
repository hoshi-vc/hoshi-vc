#!/bin/bash

# Copyright 2023 Hoshi-VC Developer <hoshi-vc@o137.net>.
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.

set -e

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
ROOT="$(dirname "$(dirname "$(dirname "$DIR")")")"

echo "Collecting data/datasets..."
tar -cf - -C "$ROOT/data" datasets | pv >"$DIR/datasets.tar"

echo "Collecting data/feats..."
tar -cf - -C "$ROOT/data" feats | pv >"$DIR/feats.tar"

echo "Collecting data/vocoder..."
tar -cf - -C "$ROOT/data" vocoder | pv >"$DIR/vocoder.tar"

echo "Collecting data/attempt07-stage1..."
tar -cf - -C "$ROOT/data" attempt07-stage1 | pv >"$DIR/attempt07-stage1.tar"

echo "Building docker image..."
docker image build -t hoshivc/trainer:full "$DIR"

echo "Pushing docker image..."
docker image push hoshivc/trainer:full

echo "Cleaning up..."
rm "$DIR/datasets.tar"
rm "$DIR/feats.tar"
rm "$DIR/vocoder.tar"
rm "$DIR/attempt07-stage1.tar"
