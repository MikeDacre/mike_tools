#!/bin/env python3
dsript="""\
#===============================================================================
#
#                FILE: selective-backup.py
#
#               USAGE: ./selective-backup.py [-o] [-l] [-r] file1 [file2]...[folder1...
#
#         DESCRIPTION: Duplicate a directory to a given location using only hard links
#        REQUIREMENTS: python3
#              AUTHOR: Michael D Dacre, mike.dacre@gmail.com
#        ORGANIZATION: Stanford University
#             LICENSE: Open-Source - No License
#             VERSION: 1.1
#             CREATED: 2013-02-27 03:09:08 PM
#            MODIFIED: 2013-04-30 14:59:19
#===============================================================================
"""

import os, re, sys, subprocess, argparse

find_parse = re.compile(r'/^\./\/home/')

### User Variables ###

# Allowed locations is a list of root file systems where users can backup.
# This script requires that all backup trees are created on a single partition.
# Right now this is only set for root level mounts, you need to edit the rootdir
# regex to change this.
allowed_locations = ['home', 'science']

# Backup paths.  This will be created in the root directory of the allowed location
#regular_backup_path = 'backup-offsite'
offsite_backup_path = 'backup-offsite'


### Argument Parsing ###

parser = argparse.ArgumentParser( formatter_class=argparse.RawDescriptionHelpFormatter,
                                                                    description=' '.join([dsript,"""
This script works only in either the""", ' or '.join(allowed_locations), """directories.
All other files are already robustly backed up.

To use simply provide the script with a list of files and folders
you would like backed up.

The default backup location is a local external disk.  If you wish
your backup to be offsite, then specify the '-o' option.

Note: Offsite backups use crashplan and are quite safe.

For more information see:
http://fraser-server.stanford.edu/wiki/index.php/Using_the_Server#Backups

Important arguments:\n"""]) )

#parser.add_argument('-o', dest='offsite', action='store_true', help="Use offsite backup location (crashplan)")
parser.add_argument('-l', dest='listfiles', action='store_true', help="List your backed up files (might be very long)")
parser.add_argument('-r', dest='rmbackup', action='store_true', help="Remove selected files from the backup")
parser.add_argument('file', type=str, nargs='*', help="Files or Folders to back up")
args = parser.parse_args()

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

        #print("Locally backed up files:")
        #os.chdir('/'.join(['', location, regular_backup_path]))
        #filelist = subprocess.check_output(['find', '.', '-user', username]).decode('utf-8').split('\n')
        #for listfile in filelist:
            #print( re.sub(r'^\.', ''.join([r'/', location]), listfile) )

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

# vim:set ts=4 sw=4 et:
