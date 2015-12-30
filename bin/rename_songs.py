#!/usr/bin/env python3
import mutagen.mp3
import sys
import shutil
for i in sys.argv[1:]:
    m = mutagen.mp3.Open(i)
    title = m.__dict__['tags']['TIT2'].text[0]
    track = m.__dict__['tags']['TRCK'].text[0].split('/')[0]
    shutil.move(i, '{0:0>2d} {1}.mp3'.format(int(track), title))
