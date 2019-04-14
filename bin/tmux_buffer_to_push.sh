#!/bin/bash

source ~/.profile

if [ -n "$TMUX" ]; then
    tmux show-buffer | ~/mike_tools/bin/notify_me.py
fi
