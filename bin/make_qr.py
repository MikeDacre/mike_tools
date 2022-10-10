#!/usr/bin/env python

# import modules
import os
import qrcode
import argparse
from PIL import Image

# Initialize parser
parser = argparse.ArgumentParser(description = "Create a QR code with central image")

# Adding optional argument
parser.add_argument("-i", "--image", help = "Use image for QR code")
parser.add_argument("-c", "--color", default="Green", help = "QR code color")
parser.add_argument("url", help = "URL to convert")
parser.add_argument("out", help = "file to write qr to (png)")

# Read arguments from command line
args = parser.parse_args()

# taking image which user wants
# in the QR code center
self_dir = os.path.dirname(os.path.abspath(__file__))
if args.image:
    logo_link = os.path.abspath(args.image)
else:
    logo_link = os.path.join(self_dir, 'Logo_Color_Small.jpg')

logo = Image.open(logo_link)

# taking base width
basewidth = 65

# adjust image size
wpercent = (basewidth/float(logo.size[0]))
hsize = int((float(logo.size[1])*float(wpercent)))
logo = logo.resize((basewidth, hsize))
QRcode = qrcode.QRCode(
    error_correction=qrcode.constants.ERROR_CORRECT_H
)

# adding URL or text to QRcode
QRcode.add_data(args.url)

# generating QR code
QRcode.make()

# taking color name from user

# adding color to QR code
QRimg = QRcode.make_image(
    fill_color=args.color, back_color="white"
).convert('RGB')

# set size of QR code
pos = ((QRimg.size[0] - logo.size[0]) // 2,
       (QRimg.size[1] - logo.size[1]) // 2)
QRimg.paste(logo, pos)

# save the QR code generated
QRimg.save(os.path.abspath(args.out))

print('QR code generated!')
