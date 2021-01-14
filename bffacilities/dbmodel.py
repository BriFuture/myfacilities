# -*- coding: utf-8 -*-

"""Description: There are two main purpose of this module:
1. provide sqlalchemy as DB to other sub modules so they can define 
their database models even the database instance is not created yet.
2. provide a function called ``createDatabase`` to attach database to 
sqlalchemy, so it can get access to these databases (which would be sqlite
or mysql database).

Author: BriFuture

Modified: 2020/05/13 20:56
解决了 mysql 中 lost connection 的错误
@refrence 
    1. https://towardsdatascience.com/use-flask-and-sqlalchemy-not-flask-sqlalchemy-5a64fafe22a4
    2. https://github.com/pallets/flask-sqlalchemy/
    3. https://docs.sqlalchemy.org/en/13/orm/contextual.html

Modified: 2020/06/19 replace property session with scoped_session_maker
"""


"""
from . import app

app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///%s' % SQLITE_DATABASE_LOC
# app.config['SQLALCHEMY_COMMIT_ON_TEARDOWN'] = True
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = True

from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy( app )
"""

__version__ = '0.1.4'
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

@as_declarative()
class _Model(object):
    # @declared_attr
    # def query():
    #     return 
    query = None
    @declared_attr
    def __tablename__(cls):
        return cls.__name__.lower()
    # id = Column(Integer, primary_key=True)
sqlalchemy.Model = _Model
orm.Query.paginate = paginate

# Database Proxy
import time
import logging, sys
import threading
class Database(object):
    """Recommond Usage: 
    ```
    db = Database(dbname)
    with db as sess:
        sess.add(Record())
        sess.commit()
    ```
    But it is compatible with flask_sqlalchemy, 
    but scopefunc must be set when construct database instance, @see create_engine
    """
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
    relationship = sqlalchemy.orm.relationship
    ForeignKey = sqlalchemy.ForeignKey
    Table = sqlalchemy.Table

    Model = _Model

    def __init__(self, dbname, logger=None, delay_engine_create=False, **kwargs):
        """db used to replace flask_sqlalchemy to provide more convinient ways to manage app
        @pool_recycle 默认的刷新时间为 10 分钟
        @Session use Session instead session property to get scoped session, 
            but scopefunc must be set before Session could be used
        @see createEngine

        kwargs scopefunc
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
        self._refresh = kwargs['pool_recycle'] - 1
        
        self.create_all = _Model.metadata.create_all
        self.drop_all = _Model.metadata.create_all
        if not delay_engine_create:
            self.create_engine(dbname, **kwargs)

    def create_engine(self, dbname, **kwargs):
        """::kwargs:: 
        scopefunc
            if flask is used, `scopefunc=_app_ctx_stack.__ident_func__` could be passed in kwargs
        """
        scopefunc = kwargs.pop('scopefunc', None)
        engine = sqlalchemy.create_engine(dbname, **kwargs)
        _Model.metadata.bind = engine
        
        global _SessionFactory
        _SessionFactory.configure(bind=engine)
        # _Session is the scoped session maker
        self.Session = orm.scoped_session(_SessionFactory, scopefunc=scopefunc)
        # used for compatibility with flask_sqlalchemy, session is scoped
        self.session = self.Session
        _Model.query = self.Session.query_property()

        self.engine = engine
        # self._lastUpdate = time.time()
        self._sessions = {}
        self.singleSession = dbname.startswith("sqlite")
        # with self as sess:
        #    sess.execute("SET session wait_timeout=5;")
        #    res = sess.execute("SELECT now();").fetchall()
        # self.logger.debug(res)


    def refreshSession(self, id = 0):
        """@deprecated 使用 mysql 时需要定时刷新 session
        check mysql wait_timeout:
        `SHOW VARIABLES LIKE "%wait%"`，如果result 小于 3600，设置 set wait_timeout=7200; 或者更大的数值。
        """
        return
        try:
            if id in self._sessions:
                self.logger.info('[Close] try close last session ')
                self._sessions[id].close()
                self._sessions[id].remove()
        except:
            # self._sessions[id].rollback()
            pass
        finally:
            self.logger.info('[create] now database session')
            self._sessions[id] = self.Session()
            # res = self._sessions[id].execute("select now();").fetchall()
    @property
    def _session(self):
        """compat with flask_sqlalchemy, not recommand 
        Try use 
        ```
        with database as  sess:
            sess.insert...
        ```
        """
        id = 0 if self.singleSession else threading.get_ident()
        sess = self._sessions.get(id, None)
        if self.valid_session(sess):
            return sess
        self.refreshSession(id)
        return self._sessions[id]

    def valid_session(self, sess):
        # now = time.time()
        # if now - self._lastUpdate > self.refresh:
        #     return False
        try:
        # Try to get the underlying session connection, If you can get it, its up
            connection = sess.connection()
            return True
        except:
            return False
        return True
    def close(self):
        self.Session.remove()

    def __enter__(self):
        self._scopesession = self.Session()
        return self._scopesession

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._scopesession.close()
        if exc_type:
            self.logger.warning(f"Type: {exc_type}, value: {exc_val}, {exc_tb}")


# Database Proxy
def createDatabase(dbname, **kwargs):
    """create instance of sqlalchemy Database
    """
    db = Database(dbname, **kwargs)
    logger = logging.getLogger('db')
    logger.warning("[DB] Use Class Database directly")
    return db