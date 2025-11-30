#!/bin/bash
export GIT_SRC=$(pwd)
export PYTHONPATH=$PYTHONPATH:${GIT_SRC}/pmpiv/pmpiv
export PYTHONPATH=$PYTHONPATH:${GIT_SRC}/pmpiv/pims
export PYTHONPATH=$PYTHONPATH:${GIT_SRC}/pmpiv/trackpy

echo $PYTHONPATH