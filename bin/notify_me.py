#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Send notification to email and push bullet.
"""
import os
import sys
import json
import argparse
from email.mime.text import MIMEText
from subprocess import Popen, PIPE
import requests

HOST = os.environ.get('HOSTNAME')
PUSH_API = "https://api.pushbullet.com/v2/pushes"
PUSH_KEY = os.environ.get('PUSH_BULLET_KEY')
SEND_ADDR = os.environ.get('EMAIL_ADDRESS')


def send_email(subject, message):
    """Send a simple sendmail email."""
    msg = MIMEText(message)
    msg["To"] = SEND_ADDR
    msg["Subject"] = subject
    p = Popen(["sendmail", SEND_ADDR], stdin=PIPE)
    print(msg.as_string())
    p.communicate(msg.as_string().encode())


def notify_push_bullet(subject, message):
    """Send a Push Bullet Notification."""
    headers = {'Content-Type':'application/json', 'Access-Token': PUSH_KEY}
    data = {"title": subject, "body": message, "type": "note"}
    r = requests.post(PUSH_API, headers=headers, data=json.dumps(data))
    if not r.ok:
        sys.stderr.write('Push Bullet Request Failed.\n{0}\n'.format(r))


def main(argv=None):
    """Notify email and push bullet."""
    if not argv:
        argv = sys.argv[1:]

    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)

    # Positional arguments
    parser.add_argument('message', help="Message to send.")

    # Optional flags
    parser.add_argument(
        '-s', '--subject', default='Notification from {0}'.format(HOST),
        help="Subject of message"
    )
    parser.add_argument('--push-key', help="Override PUSH_BULLET_KEY variable")
    parser.add_argument('--email-addr', help="Override EMAIL_ADDRESS variable")

    parser.add_argument(
        '--skip-push', action='store_true', help="Don't send push"
    )
    parser.add_argument(
        '--send-email', action='store_true', help="Also send email"
    )

    args = parser.parse_args(argv)

    if args.email_addr:
        global SEND_ADDR
        SEND_ADDR = args.email
    if args.push_key:
        global PUSH_KEY
        PUSH_KEY = args.push_key

    if not args.skip_push:
        notify_push_bullet(args.subject, args.message)
    if not args.skip_email:
        send_email(args.subject, args.message)


if __name__ == '__main__' and '__file__' in globals():
    sys.exit(main())
