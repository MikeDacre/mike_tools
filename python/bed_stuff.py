"""
Filter all snps in a bed or vcf file, return those in region of another bed.

============================================================================

        AUTHOR: Michael D Dacre, mike.dacre@gmail.com
  ORGANIZATION: Stanford University
       LICENSE: MIT License, property of Stanford, use as you wish
       VERSION: 0.1
       CREATED: 2016-51-16 15:03
 Last modified: 2016-03-16 18:25

   DESCRIPTION:

         USAGE: Import as a module or run as a script

============================================================================
"""
import os
import re
import bz2
import gzip
import sqlite3
import bed_lookup
import logme


def sqlite_from_bed(bedfile):
    """Make an sqlite file from a bedfile."""
    db_name = bedfile if bedfile.endswith('.db') else bedfile + '.db'
    # Check if the alternate db exists if db doesn't exist
    exists = False
    if not os.path.exists(db_name):
        if bedfile.endswith('.gz'):
            alt_path = '.'.join(bedfile.split('.')[:-1]) + '.db'
        else:
            alt_path = bedfile + '.gz' + '.db'
        if os.path.exists(alt_path):
            db_name = alt_path
            exists = True
    else:
        exists = True

    # If the database already exists, use it
    if exists:
        logme.log('Using existing db, if this is ' +
                    'not what you want, delete ' + db_name,
                    level='info')
        conn = sqlite3.connect(db_name)
        c = conn.cursor()
        return

    # Create an sqlite database from bed file
    logme.log('Creating sqlite database, this ' +
                'may take a long time.', level='info')
    conn = sqlite3.connect(db_name)
    c = conn.cursor()

    with open_zipped(bedfile) as infile:
        for line in infile:
            f = line.rstrip().split('\t')
            if len(f) < 4:
                continue
            # Check if db exists and create if it does
            expr = ("SELECT * FROM sqlite_master WHERE name = '{}' " +
                    "and type='table';").format(f[0])
            c.execute(expr)
            if not c.fetchall():
                exp = ("CREATE TABLE '{}' (name text, start int, " +
                        "end int);").format(f[0])
                c.execute(exp)
                conn.commit()
            expr = ("INSERT INTO '{}' VALUES " +
                    "('{}','{}','{}')").format(f[0], f[3], f[1], f[2])
            c.execute(expr)
        conn.commit()
        # Create indicies
        c.execute('''SELECT name FROM sqlite_master WHERE type='table';''')
        for i in c.fetchall():
            exp = ("CREATE INDEX '{0}_start_end' ON '{0}' " +
                    "(start, end)").format(i[0])
            c.execute(exp)
            conn.commit()

    return db_name


def bed_interator(bedfile):
    """Generator to iterate through bed. Returns: name, chr, pos,
    strand.
    """
    with open_zipped(bedfile) as fin:
        for line in fin:
            fields = line.rstrip().split('\t')
            yield(fields[3], fields[0], int(fields[1]), fields[5])


def gtf_interator(gtffile, feature=None):
    """Generator to iterate through gtf. Returns: name_feature, chr, pos,
    strand.

    NOTE: Position is base 1

    :feature: Filter on this feature.
    """
    search = re.compile(r'gene_id "([^"]+)"')
    with open_zipped(gtffile) as fin:
        for line in fin:
            fields = line.rstrip().split('\t')
            if feature:
                if not fields[2] == feature:
                    continue
            yield('{}_{}'.format(search.findall(fields[8])[0], fields[2]),
                  fields[0], int(fields[3])+1, fields[5])


def open_zipped(infile, mode='r'):
    """ Return file handle of file regardless of zipped or not
        Text mode enforced for compatibility with python2 """
    mode   = mode[0] + 't'
    p2mode = mode
    if hasattr(infile, 'write'):
        return infile
    if isinstance(infile, str):
        if infile.endswith('.gz'):
            return gzip.open(infile, mode)
        if infile.endswith('.bz2'):
            if hasattr(bz2, 'open'):
                return bz2.open(infile, mode)
            else:
                return bz2.BZ2File(infile, p2mode)
        return open(infile, p2mode)

