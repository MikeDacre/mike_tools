#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Get push bullet notification.
"""
import os
import sys
import json
import argparse
from subprocess import Popen, PIPE

# Doesn't work on python2
if sys.version_info.major != 3:
    sys.stderr.write("Doesn't work on python2\n")
    sys.exit(1)

try:
    import requests
    use_alternate_method = False
except ImportError:
    import urllib
    import urllib.request
    import urllib.parse
    use_alternate_method = True

HOST = os.environ.get('HOSTNAME')
PUSH_API = "https://api.pushbullet.com/v2/pushes"
PUSH_KEY = os.environ.get('PUSH_BULLET_KEY')


def send_get(data):
    """Use urllib to send instead."""
    # Build basic stuff
    handler = urllib.request.HTTPHandler()
    opener = urllib.request.build_opener(handler)
    # Prepare the request
    data = urllib.parse.urlencode(data)
    long_url = PUSH_API + '?' + data
    request = urllib.request.Request(long_url)
    request.add_header('Access-Token', PUSH_KEY)
    request.add_header('Conent-Type', 'application/json')
    request.get_method = lambda: "GET"
    # Send it
    try:
        connection = opener.open(request)
    except urllib.request.HTTPError as err:
        connection = err
    return connection


def get_push_bullet():
    """Get a Push Bullet Notification."""
    headers = {'Content-Type': 'application/json', 'Access-Token': PUSH_KEY}
    req = {'active': 'true', 'limit': '1'}
    if use_alternate_method:
        connection = send_get(req)
        code = connection.code
    else:
        r = requests.get(PUSH_API, params=req, headers=headers)
        code = r.status_code
    if not code == 200:
        sys.stderr.write(
            'Push Bullet request failed with code {0}\n'.format(code)
        )
        return None
    if use_alternate_method:
        pushes = json.loads(connection.read().decode())
    else:
        pushes = json.loads(r.content.decode())
    push = pushes['pushes'][0]
    message = push['body']
    return message


def main(argv=None):
    """Notify email and push bullet."""
    if not argv:
        argv = sys.argv[1:]

    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)

    conf = parser.add_argument_group('config')
    conf.add_argument('--push-key', help="Override PUSH_BULLET_KEY variable")

    args = parser.parse_args(argv)

    if args.push_key:
        global PUSH_KEY
        PUSH_KEY = args.push_key

    if not PUSH_KEY:
        sys.stderr.write('No Push Bullet API (PUSH_BULLET_KEY)\n')
        return 1

    message = get_push_bullet()

    sys.stdout.write(message)
    return 0


if __name__ == '__main__' and '__file__' in globals():
    sys.exit(main())
