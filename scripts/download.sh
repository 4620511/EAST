#!/bin/bash

set -eu

if ! type -p gdown >/dev/null; then
	echo "gdown not found on the system" >&2
	exit 1
fi

SCRIPTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
WEIGHTS_DIR="$SCRIPTS_DIR/../weights"
WEIGHTS_PATH="$WEIGHTS_DIR/east_vgg16.pth"
URL="https://drive.google.com/file/d/1AFABkJgr5VtxWnmBU3XcfLJvpZkC2TAg/view"

mkdir -p "$WEIGHTS_DIR"
gdown --fuzzy --output "$WEIGHTS_PATH" "$URL"
