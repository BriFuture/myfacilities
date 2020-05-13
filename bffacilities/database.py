
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
from sqlalchemy.ext.declarative import declarative_base
Model = declarative_base()
sqlalchemy.Model = Model
# Database Proxy
import time
class _Database():
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
    def __init__(self, Session, refresh = 300):
        """默认的刷新时间为 5 分钟
        """
        Model.query = Session.query_property()
        self.Model = Model
        self._Session = Session
        self._lastUpdate = time.time()
        self._session = Session()
        self.refresh = refresh - 1
    @property
    def session(self):
        now = time.time()
        if now - self._lastUpdate > self.refresh:
            self._lastUpdate = now
            try:
                self._session.close()
            except:
                pass
            finally:
                self._session = self._Session()
                return self._session
        else:
            return self._session

# Database Proxy
def createDatabase(dbname, **kwargs):
    """create instance of sqlalchemy Database
    """
    if 'pool_recycle' not in kwargs:
        kwargs['pool_recycle'] = 600
    engine = sqlalchemy.create_engine(dbname, **kwargs)
    session_factory = orm.sessionmaker(bind=engine)
    _Session = orm.scoped_session(session_factory )
    Model.metadata.bind = engine
    db = _Database(_Session, refresh=kwargs['pool_recycle'])
    db.engine = engine
    db.create_all = Model.metadata.create_all
    orm.Query.paginate = paginate
    return db