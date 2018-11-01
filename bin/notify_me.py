#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Send notification to email and push bullet.
"""
import os
import sys
import json
import argparse
import bz2 as _bz2
import gzip as _gzip
from email.mime.text import MIMEText
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
SEND_ADDR = os.environ.get('EMAIL_ADDRESS')


def send_post(data):
    """Use urllib to send instead."""
    # Build basic stuff
    handler = urllib.request.HTTPHandler()
    opener = urllib.request.build_opener(handler)
    # Prepare the request
    data = urllib.parse.urlencode(data).encode()
    request = urllib.request.Request(PUSH_API, data=data)
    request.add_header('Access-Token', PUSH_KEY)
    request.add_header('Conent-Type', 'application/json')
    request.get_method = lambda: "POST"
    # Send it
    try:
        connection = opener.open(request)
    except urllib.request.HTTPError as err:
        connection = err
    return connection.code


def send_email(subject, message):
    """Send a simple sendmail email."""
    msg = MIMEText(message)
    msg["To"] = SEND_ADDR
    msg["Subject"] = subject
    p = Popen(["sendmail", SEND_ADDR], stdin=PIPE)
    p.communicate(msg.as_string().encode())


def notify_push_bullet(subject, message):
    """Send a Push Bullet Notification."""
    headers = {'Content-Type': 'application/json', 'Access-Token': PUSH_KEY}
    data = {"title": subject, "body": message, "type": "note"}
    if use_alternate_method:
        code = send_post(data)
    else:
        r = requests.post(PUSH_API, headers=headers, data=json.dumps(data))
        code = r.status_code
    if not code == 200:
        sys.stderr.write(
            'Push Bullet request failed with code {0}\n'.format(code)
        )


def open_zipped(infile, mode='r'):
    """return file handle of file regardless of compressed or not.

    also returns already opened files unchanged, text mode automatic for
    compatibility with python2.
    """
    # return already open files
    if hasattr(infile, 'write'):
        return infile
    # make text mode automatic
    if len(mode) == 1:
        mode = mode + 't'
    # refuse to handle non-strings that aren't files.
    if not isinstance(infile, str):
        raise ValueError("i cannot open a filename that isn't a string.")
    # treat '-' appropriately
    if infile == '-':
        if 'w' in mode:
            return sys.stdout
        return sys.stdin
    # if possible open zipped files
    if infile.endswith('.gz'):
        return _gzip.open(infile, mode)
    if infile.endswith('.bz2'):
        if hasattr(_bz2, 'open'):
            return _bz2.open(infile, mode)
        return _bz2.bz2file(infile, mode)
    # fall back on regular open
    return open(infile, mode)


def main(argv=None):
    """Notify email and push bullet."""
    if not argv:
        argv = sys.argv[1:]

    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)

    # Positional arguments
    parser.add_argument('message', nargs='?', help="Message to send.")

    # Optional flags
    parser.add_argument(
        '-s', '--subject', default='Notification from {0}'.format(HOST),
        help="Subject of message"
    )

    method = parser.add_argument_group('methods')
    method.add_argument(
        '--skip-push', action='store_true', help="Don't send push"
    )
    method.add_argument(
        '--send-email', action='store_true', help="Also send email"
    )

    conf = parser.add_argument_group('config')
    conf.add_argument('--push-key', help="Override PUSH_BULLET_KEY variable")
    conf.add_argument('--email-addr', help="Override EMAIL_ADDRESS variable")

    args = parser.parse_args(argv)

    if args.email_addr:
        global SEND_ADDR
        SEND_ADDR = args.email_addr
    if args.push_key:
        global PUSH_KEY
        PUSH_KEY = args.push_key

    if args.message:
        if isinstance(args.message, str):
            if os.path.isfile(args.message):
                message = open_zipped(args.message).read()
            elif args.message == '-':
                message = sys.stdin.read()
            else:
                message = args.message
        else:
            try:
                message = open_zipped(args.message).read()
            except:
                message = args.message
    else:
        message = sys.stdin.read()
    #  message = args.message if args.message else sys.stdin.read()

    sent = False
    if not args.skip_push and PUSH_KEY:
        notify_push_bullet(args.subject, message)
        sent = True
    if args.send_email and SEND_ADDR:
        send_email(args.subject, message)
        sent = True
    if not sent:
        sys.stderr.write('No destinations specified\n')
        return 2


if __name__ == '__main__' and '__file__' in globals():
    sys.exit(main())
