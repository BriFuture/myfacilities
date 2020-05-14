"""Description: There are two main purpose of this module:
1. provide sqlalchemy as DB to other sub modules so they can define 
their database models even the database instance is not created yet.
2. provide a function called ``createDatabase`` to attach database to 
sqlalchemy, so it can get access to these databases (which would be sqlite
or mysql database).

Author: BriFuture

Modified: 2020/05/13 20:56
解决了 mysql 中 lost connection 的错误
"""


"""
from . import app

app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///%s' % SQLITE_DATABASE_LOC
# app.config['SQLALCHEMY_COMMIT_ON_TEARDOWN'] = True
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = True

from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy( app )
"""

__version__ = '0.1.2'
import math

def paginate(self, page=None, per_page=None, to_dict=True):
    """
    分页函数
    :param self:
    :param page:
    :param per_page:
    :return:
    """
    if page is None:
        page = 1

    if per_page is None:
        per_page = 20

    items = self.limit(per_page).offset((page - 1) * per_page).all()

    if not items and page != 1:
        return {'total': 0, 'page': page, 'error': 'no such items'}
        
    if page == 1 and len(items) < per_page:
        total = len(items)
    else:
        total = self.order_by(None).count()
    
    if to_dict:
        ditems = [item.to_dict() for item in items]
    else:
        ditems = items

    return {
        'page': page, 
        'per_page': per_page, 
        'total': total, 
        'items': ditems
    }
    # return Pagination(self, page, per_page, total, items)

import sqlalchemy
from sqlalchemy import orm
from sqlalchemy.ext.declarative import declarative_base, as_declarative, declared_attr
# Model = declarative_base()
_SessionFactory = orm.sessionmaker()
_Session = None
@as_declarative()
class Model(object):
    # @declared_attr
    # def query():
    #     return 
    query = None
    @declared_attr
    def __tablename__(cls):
        return cls.__name__.lower()
    # id = Column(Integer, primary_key=True)
sqlalchemy.Model = Model
orm.Query.paginate = paginate

# Database Proxy
import time
import logging, sys
class Database():
    Column = sqlalchemy.Column
    Integer = sqlalchemy.Integer
    SmallInteger = sqlalchemy.SmallInteger
    BigInteger = sqlalchemy.BigInteger
    Boolean = sqlalchemy.Boolean
    Enum = sqlalchemy.Enum
    Float = sqlalchemy.Float
    Interval = sqlalchemy.Interval
    Numeric = sqlalchemy.Numeric
    PickleType = sqlalchemy.PickleType
    String = sqlalchemy.String
    Text = sqlalchemy.Text
    DateTime = sqlalchemy.DateTime
    Date = sqlalchemy.Date
    Time = sqlalchemy.Time
    Unicode = sqlalchemy.Unicode
    UnicodeText = sqlalchemy.UnicodeText
    LargeBinary = sqlalchemy.LargeBinary
    # MatchType = sqlalchemy.MatchType
    # SchemaType = sqlalchemy.SchemaType
    ARRAY = sqlalchemy.ARRAY
    BIGINT = sqlalchemy.BIGINT
    BINARY = sqlalchemy.BINARY
    BLOB = sqlalchemy.BLOB
    def __init__(self, dbname, logger=None, **kwargs):
        """默认的刷新时间为 5 分钟
        """
        if logger is None:
            logger = logging.getLogger('db')
            formatter = logging.Formatter(
                "[%(asctime)s %(levelname)s] - %(funcName)s -  %(message)s")
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(formatter)
            logger.setLevel(logging.DEBUG)
            logger.addHandler(console_handler)
            self.logger = logger

        if 'pool_recycle' not in kwargs:
            kwargs['pool_recycle'] = 600
        # kwargs['pool_recycle'] = 5 # TEST
        self.refresh = kwargs['pool_recycle'] - 1
        engine = sqlalchemy.create_engine(dbname, **kwargs)
        
        global _SessionFactory, _Session
        _SessionFactory.configure(bind=engine)
        _Session = orm.scoped_session(_SessionFactory)
        Model.query = _Session.query_property()
        Model.metadata.bind = engine

        self.engine = engine
        self.Model = Model
        self.Session = _Session
        self._lastUpdate = time.time()
        self._session = _Session()
        # print('test', Model.query, self._session)
        # self._session.execute("SET session wait_timeout=5;")
        # res = self.session.execute("SELECT now();").fetchall()
        # self.logger.debug(res)
        self.create_all = Model.metadata.create_all

    def refreshSession(self):
        """使用 mysql 时需要定时刷新 session
        check mysql wait_timeout:
        `SHOW VARIABLES LIKE "%wait%"`，如果result 小于 3600，设置 set wait_timeout=7200; 或者更大的数值。
        """
        try:
            self.logger.info('[Close] try close last session ')
            self._session.close()
        except:
            pass
        finally:
            self.logger.info('[create] now database session')
            self._session = self.Session()
            # res = self._session.execute("select now();").fetchall()
    @property
    def session(self):
        now = time.time()
        if now - self._lastUpdate > self.refresh:
            # self.logger.debug(f'last session is invalid now {now - self._lastUpdate}')
            self._lastUpdate = now
            self.refreshSession()
            return self._session
        else:
            return self._session

# Database Proxy
def createDatabase(dbname, **kwargs):
    """create instance of sqlalchemy Database
    """
    db = Database(dbname, **kwargs)
    return db