#!/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8 tabstop=4 expandtab shiftwidth=4 softtabstop=4
#
# Copyright © Mike Dacre <mike.dacre@gmail.com>
#
# Distributed under terms of the MIT license
"""\
===============================================================================

                FILE: selective-backup.py

              AUTHOR: Michael D Dacre, mike.dacre@gmail.com
        ORGANIZATION: Stanford University
             LICENSE: MIT License

             CREATED: 2013-02-27 03:09:08 PM
            MODIFIED: 2013-04-30 14:59:19

               USAGE: ./selective-backup.py [-l] [-r] file1 [file2]...[folder1...

         DESCRIPTION: Create a hard link tree for use with backup software.
                      Currently designed to work with CrashPlanPro
                      Number of allowed locations is specified in allowed_locations
                      variable; this prevents running from non-standard locations.

                      The script can create and delete backups, as well as list all
                      user backups in a given location.

                NOTE: Does NOT actually do ANY BACKUPS. Simply facilitates backups

             VERSION: 0.2
        REQUIREMENTS: python3
===============================================================================

For more information see:
http://fraser-server.stanford.edu/wiki/index.php/Using_the_Server#Backups
"""

import os, re, sys, subprocess, argparse

### User Variables ###

# Allowed locations is a list of root file systems where users can backup.
# This script requires that all backup trees are created on a single partition.
# Right now this is only set for root level mounts, you need to edit the rootdir
# regex to change this.
allowed_locations = ['home', 'science']

# Backup paths.  This will be created in the root directory of the allowed location
regular_backup_path = 'backup-offsite'
offsite_backup_path = 'backup-offsite'

def _get_args():
    """Command Line Argument Parsing"""
    import argparse, sys

    helpstring = '\n'.join([__doc__,
                 "This script works only in either the",
                 ' or '.join(allowed_locations), "\ndirectories.\n"
    All other files are already robustly backed up.


    parser = argparse.ArgumentParser(
                 description=__doc__,
                 formatter_class=argparse.RawDescriptionHelpFormatter)

    # Required Files
    parser.add_argument('file', type=str, nargs='*', help="Files or Folders to back up")

    # Location selection
    #parser.add_argument('-d', dest='destination',
            #help="Use offsite backup location (crashplan)")

    # Other options
    parser.add_argument('-l', dest='listfiles', action='store_true',
            help="List your backed up files (might be very long)")
    parser.add_argument('-r', dest='rmbackup', action='store_true',
            help="Remove selected files from the backup")

    # Optional Arguments
    parser.add_argument('-v', help="Verbose output")

    args = parser.parse_args()
    return parser

# Main function for direct running
def main():
    """Run directly"""
    # Get commandline arguments
    parser = _get_args()
    args = parser.parse_args()

    find_parse = re.compile(r'/^\./\/home/')

    ### Error Checking and Variable Setting ###

    # Set location for backup.  In this version, only /science and /home are valid starting
    # locations.  If '-o' is specified, then put in offsite location.

    # Default exit number is 0
    errorno = 0

    # List files and exit if requested
    if args.listfiles:
        username = subprocess.check_output('whoami').rstrip()

        # Display all listed files
        for location in allowed_locations:
            print(' '.join(["\n", location, "directory\n"]))

            print("Locally backed up files:")
            os.chdir('/'.join(['', location, regular_backup_path]))
            filelist = subprocess.check_output(['find', '.', '-user', username]).decode('utf-8').split('\n')
            for listfile in filelist:
                print( re.sub(r'^\.', ''.join([r'/', location]), listfile) )

            print("\nOffsite backed up files:")
            os.chdir('/'.join(['', location, offsite_backup_path]))
            filelist = subprocess.check_output(['find', '.', '-user', username]).decode('utf-8').split('\n')
            for listfile in filelist:
                print( re.sub(r'^\.', ''.join([r'/', location]), listfile) )

        print("\nYou can delete these files and directories manually with `rm -rf`,")
        print("or you can use the -r option with this script.")
        sys.exit()

    else:
        if not args.file:
            parser.print_help()
            print("\nError: You must specify at least one file to back up\n")
            sys.exit(1001 )

    if args.rmbackup:
        print("You have chosen to remove your backups, rather than to create new ones.")
        cont_choice = input("Are you sure you want to remove the selected files? [y/N] ")
        if not cont_choice.lower() == 'y':
            print("Aborting...")
            sys.exit(1002)

        for file in args.file:

            # Get root folder
            filepath = os.path.abspath(file)
            rootdir, filesubpath, filename = re.match('/([^/]+)/(.*)/([^/]+)$', filepath).groups()

            # Make sure location is allowed
            if rootdir not in allowed_locations:
                print( ' '.join(["Error:\n", file, "is not in an allowed location. For allowed locations, run with '-h'\n"]), file=sys.stderr)
                errorno = errorno + 1
                continue

            ## Set backup location
            #if args.offsite:
                #backup_path = offsite_backup_path
            #else:
                #backup_path = regular_backup_path
            backup_path = offsite_backup_path

            backuplocation = '/'.join(['', rootdir, backup_path, filesubpath])
            backupdestination = '/'.join([backuplocation, filename])

            # Make sure backup exists
            if os.path.exists(backupdestination):
                # Delete file
                errout = subprocess.check_call(['rm', '-rf', backupdestination])
                if errout:
                    print("\nThe command:\n")
                    print(' '.join(["rm -rf", backupdestination]))
                    print(' '.join(["failed with the following error:", errout]))
                    errorno = errorno + 1
                else:
                    print(' '.join([filename, "successfully removed from the backup"]))
                # Check parent directories
                dirs = filesubpath.split('/')
                while dirs:
                    rmpath = '/'.join(['', rootdir, backup_path,
                                        '/'.join(dirs) ])
                    if not os.listdir(rmpath):
                        errout = subprocess.check_call(['rm', '-rf', rmpath])
                        if errout:
                            print("\nThe command:\n")
                            print(' '.join(["rm -rf", backupdestination]))
                            print(' '.join(["failed with the following error:", errout]))
                            errorno = errorno + 1
                            break
                        dirs.pop()
                    else:
                        break

            else:
                print(' '.join([backupdestination, "does not exist in the backup, no point removing"]))

    ### Backup Script ###
    else:
        for file in args.file:

            # Get root folder
            filepath = os.path.abspath(file)
            rootdir, filesubpath, filename = re.match('/([^/]+)/(.*)/([^/]+)$', filepath).groups()

            # Check if file exists
            if not os.path.exists(filepath):
                print( ' '.join(["Error:\n", file, "does not exist, not backing up\n"]), file=sys.stderr)
                errorno = errorno + 1
                continue

            # Make sure location is allowed
            if rootdir not in allowed_locations:
                print( ' '.join(["Error:\n", file, "is not in an allowed location. For allowed locations, run with '-h'\n"]), file=sys.stderr)
                errorno = errorno + 1
                continue

            ## Set backup location
            #if args.offsite:
                #backup_path = offsite_backup_path
            #else:
                #backup_path = regular_backup_path

            ## Only offsite now
            backup_path = offsite_backup_path

            backuplocation = '/'.join(['', rootdir, backup_path, filesubpath])
            backupdestination = '/'.join([backuplocation, filename])

            # Check if file already backed up
            if os.path.exists(backuplocation):
                if os.path.exists(backupdestination):
                    if os.path.isdir(backupdestination):
                        print(' '.join([file, "Already backed up at:\n", backupdestination, '\n']), file=sys.stderr)
                        errorno = errorno + 1
                        continue
                    elif subprocess.check_output(['/usr/bin/stat', '-c', '%d:%i', filepath]) == subprocess.check_output(['/usr/bin/stat', '-c', '%d:%i', backupdestination]):
                        print(' '.join([file, "Already backed up at:\n", backupdestination, '\n']), file=sys.stderr)
                        errorno = errorno + 1
                        continue
                    else:
                        print(' '.join([backupdestination, "already exists, but it is not the same as", filepath]))
                        overwriteresponse = input('Do you want to overwrite the file? [y/N]')

                        if overwriteresponse.lower() == 'y' :
                            subprocess.check_call(['rm', '-rf', backupdestination])
                        else:
                            continue

            # Create backup destination
            else:
                subprocess.check_call(['mkdir', '-p', backuplocation])

            # Run copy command
            errout = subprocess.check_call(['cp', '-al', filepath, backuplocation])
            if errout:
                print(' '.join(["Error:\n", "The command:\n", "cp -al", filepath, backuplocation,
                                "\n", "Failed with error number", errout, "\n",
                                "The file:", filepath, "was probably not backed up.\n",
                                "Please try again.\n"]))
                errorno = errorno + 1
            else:
                print(' '.join([file, "backed up successfully"]))


    ### Error Processing ###

    if errorno:
        print("\nScript completed successfully, but there were errors with the files you chose.  Please see stderr output for details")
        sys.exit(errorno)

# The end
if __name__ == '__main__':
    main()
