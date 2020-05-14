# -*- coding: utf-8 -*-
"""Description: This module is used to provide email feature to all
other sub modules so that they can send email by a simple api.

Author: BriFuture

Modified: 2019/03/17 12:35
"""


__version__ = '0.1.2'

import sqlalchemy
from bffacilities.utils import createLogger
from datetime import datetime
logger = createLogger('email', stream=False)

class Email( sqlalchemy.Model ):
    __tablename__ = 'email'
    id = sqlalchemy.Column( sqlalchemy.Integer, primary_key = True, autoincrement = True )
    to = sqlalchemy.Column( sqlalchemy.String( 255 ), nullable=False )
    content = sqlalchemy.Column( sqlalchemy.Text )
    time = sqlalchemy.Column( sqlalchemy.DateTime, default=datetime.now )
    subject = sqlalchemy.Column( sqlalchemy.String( 255 ) )
    ip = sqlalchemy.Column( sqlalchemy.String( 255 ) )
    type = sqlalchemy.Column( sqlalchemy.String( 255 ) )
    remark = sqlalchemy.Column( sqlalchemy.String( 255 ) )
    session = sqlalchemy.Column( sqlalchemy.String( 255 ) )

    def to_dict(self):
        di = {
            'id': self.id,
            'to': self.to,
            'content': self.content,
            'time': self.time.timestamp(),
            'subject': self.subject,
            'ip': self.ip,
            'type': self.type,
            'remark': self.remark,
            'session': self.session,            
        }
        return di

    def __repr__(self):
        return "<Email to: {}, subject: {}, content: {}, time: {}>"\
            .format(self.to, self.subject, self.content, self.time)

from queue import Queue
import smtplib
import yaml
from email import encoders
from email.header import Header
from email.mime.text import MIMEText
from email.utils import parseaddr, formataddr


class MailSender(object):
    """``EmailSender`` is a wrapper of ``email`` module
    """
    def __init__(self, host, account, password, port, *args, **kwargs):
        self._host = host
        self._account = account
        self._password = password
        self._port = port
        self.server = smtplib.SMTP_SSL(self._host, self._port)

    def reset_server(self):
        self.server = smtplib.SMTP_SSL(self._host, self._port)

    def send(self, email: Email):
        msg = MIMEText(email.content, 'html', 'utf-8')
        msg['Subject'] = Header(email.subject, 'utf-8').encode()
        msg['From'] = self.__format_addr('SSPY-Mgr <{}>'.format(self._account))
        msg['To'] = self.__format_addr('Receiver <{}>'.format(email.to) )
        self.server.connect( self._host, self._port )
        # self.server.starttls()
        self.server.ehlo()
        self.server.login( self._account, self._password)
        self.server.sendmail( self._account, email.to, msg.as_string())
        self.server.quit()

    def __format_addr( self, s ):
        name, addr = parseaddr(s)
        return formataddr( ( Header(name, 'utf-8').encode(), addr ) )

from flask import jsonify 
def admin_getAllEmails():
    emails = Email.query.all()
    em = [e.to_dict() for e in emails]
    return jsonify({'status': 'success', 'emails': em})
from .utils import getPageArgs
def admin_getPageEmails():
    page, per_page = getPageArgs()
    emails = Email.query.paginate(page=page, per_page=per_page)
    return jsonify({'status': 'success', 'emails': emails})


class EmailManager(object):
    """Used for generating ``Email`` database record and sending email
    """
    def __init__(self, config, database, *args, **kwargs):
        # self.config = config
        email = config.email
        self.db = database
        self._test = config.debug
        self._host     = email[ 'host' ]
        if self._host == 'smtp host':
            logger.warning( '[Email] Please Place Your smtp account into config file')
            import sys
            sys.exit( 1 )
        if( self._host == None ) :
            raise ValueError

        port = email.get( 'port' ) or 465
        self.sender = MailSender(self._host, email[ 'account' ], email[ 'password' ], port)
        self._process_pending = False
        self._sendQueue = Queue()
    def registerApi(self, api):
        api.add_url_rule('/email/getAll', view_func=admin_getAllEmails, methods=['GET', 'POST'])
        api.add_url_rule('/email/getPage', view_func=admin_getPageEmails, methods=['GET', 'POST'])
        logger.info("Api registered")
        
    def add_email(self, to, subject, content, type = None ):
        """append an email object to queue, the mail will be sent at proper time,
        Empty subject or content is not allowed.
        """
        # it may be better if the email pattern is checked
        if len(to) == 0:
            logger.warning("Remote address not set in an email")
            return
        if len(subject) == 0:
            logger.warning("Empty subject found in an email")
            return
        if len(content) == 0:
            logger.warning("Empty content found in an email")
            return

        email = Email(
            to=to,
            subject=subject,
            content=content,
            type=type,
            remark="pending")
        # db.session.add(email)
        self._sendQueue.put(email)
    
    def send(self, email: Email):
        self._sendQueue.put(email)

    def checkRemain(self):
        """Check it if there exists any pending emails that need to be sent
        """
        if self._sendQueue.empty():
            self._checkPending()
            return
        email = self._sendQueue.get()
        self.__send(email, commit=True)

    def _checkPending(self):
        if self._process_pending == True:
            return
        self._process_pending = True
        emails = Email.query.filter_by(remark='pending').all()
        for e in emails:
            self.__send(e)
        self.db.session.commit()
        self._process_pending = False

    def __send(self, email: Email, commit = False):
        """Do the action of sending email, 
        ``commit`` set True to invoke db session commit.
        """
        self.db.session.add(email)
        logger.debug("Geting email and ready to send it")
        if self._test:
            email.remark = "debugNotSend"
            logger.info("Email generated but not send: {}".format(email))
        else:
            self.sender.reset_server()
            self.sender.send(email)
            email.remark = "sent"
            logger.debug("Email generated and send: {}".format(email))

        if commit:
            self.db.session.commit()

