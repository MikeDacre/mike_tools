#!/bin/bash

source ~/.profile

if [ -n "$TMUX" ]; then
    tmux set-buffer -b push "$($HOME/mike_tools/bin/get_push.py)"
fi
