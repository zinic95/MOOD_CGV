#!/usr/bin/env bash
SCRIPTPATH="$( cd "$(dirname "$0")" ; pwd -P )"

docker build -t mood_test:local "$SCRIPTPATH" -f ./Dockerfile