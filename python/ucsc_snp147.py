"""
Use SQLAlchemy to access UCSC SNP147 Table.

Method described here:
    https://gist.github.com/MikeDacre/7531d4e02052cbb8e906a084ad68e4e7

       Created: 2017-36-09 12:02
 Last modified: 2017-02-09 12:41

"""
import pandas as _pd
import sqlalchemy as _sq
from sqlalchemy import Column, Integer, String
from sqlalchemy.orm import Session as _Session
from sqlalchemy.ext.automap import automap_base as _automap_base

# Connect to the hg19 database
engine = _sq.create_engine(
    "mysql+pymysql://genome@genome-mysql.cse.ucsc.edu/{organism}?charset=utf8mb4"
    .format(organism='hg19')
)

Base = _automap_base()

class snp147(Base):
    __tablename__ = 'snp147'

    name = _sq.Column(String(length=15), primary_key=True, nullable=False)

    # The following columns do not need to be declared, the automapper will do it for
    # us. I map them anyway for my own personal reference.
    chrom      = _sq.Column(_sq.String(length=31), nullable=False)
    chromStart = _sq.Column(_sq.Integer, nullable=False)
    chromEnd   = _sq.Column(_sq.Integer, nullable=False)

# reflect the tables
Base.prepare(engine, reflect=True)

session = _Session(engine)
