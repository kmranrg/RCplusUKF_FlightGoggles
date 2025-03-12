#!/bin/bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

docker run \
    -it \
    --publish-all \
    --rm \
    --volume "${DIR}/../pyFlightGoggles:/root/pyFlightGoggles" \
    --name py_flightgoggles \
    --privileged \
    -p 8888:8888 \
    pyfg
