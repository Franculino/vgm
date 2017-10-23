import logging
import logging.handlers
import string
import os
import glob
import vgm

__all__ = ['StreamFormatter', 'LoggingDispatcher']


# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------


class StreamFormatter(logging.Formatter):
    def __init__(self, s_lt, s_ge, lthresh):
        logging.Formatter.__init__(self, s_lt)
        self._lthresh = logging.getLevelName(lthresh)
        self.s_lt = s_lt
        self.s_ge = s_ge

    def format(self, record):
        if record.levelno < self._lthresh:
            self._fmt = self.s_lt
        else:
            self._fmt = self.s_ge
        msg = logging.Formatter.format(self, record)
        return msg


# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------


class LoggingDispatcher(object):
    def __init__(self):
        self.loglevels = {'debug': logging.DEBUG,
                        'info': logging.INFO,
                        'warning': logging.WARNING,
                        'error': logging.ERROR,
                        'critical': logging.CRITICAL}
        # Note that exceptions have error-level, but require an exception handler
        self.loglist = {}
        self.logdir = os.path.join(vgm.basedir, vgm.ConfParser.get('logging', 'logdir'))
        self.pid = os.getpid()

    def create_logger(self, name):
        loglevels = self.loglevels
        confp = vgm.ConfParser
        option = name if confp.has_option('logging', name) else 'default_settings'

        level_s, level_f = string.split(confp.get('logging', option), ' ')
        delete_previous = confp.getboolean('logging', 'delete_previous')
        append_pid = confp.getboolean('logging', 'append_pid')
        if delete_previous:
            self.delete_logfiles(False, True)
            writemode = 'w'
        else:
            writemode = 'a'
        
        s_lt = confp.get('logging', 'format_s_lt', raw=True)
        s_ge = confp.get('logging', 'format_s_ge', raw=True)
        lthresh = string.upper(confp.get('logging', 'format_s_lthresh'))
        formatter_s = StreamFormatter(s_lt, s_ge, lthresh)
        formatter_f = logging.Formatter(confp.get('logging', 'format_f', raw=True))

        log = logging.getLogger(name)
        log.setLevel(loglevels['debug'])
        handler_s = logging.StreamHandler()
        handler_s.setLevel(loglevels[level_s])
        handler_s.setFormatter(formatter_s)
        log.addHandler(handler_s)
        if append_pid:
            logbasename = '%s_%i.log' % (name, self.pid)
        else:
            logbasename = '%s.log' % (name)
        logname = os.path.join(self.logdir, logbasename)
        handler_f = logging.FileHandler(logname, mode=writemode)
        handler_f.setLevel(loglevels[level_f])
        handler_f.setFormatter(formatter_f)
        log.addHandler(handler_f)
        self.loglist[logbasename] = log
        return log

    def add_logger(self, name, log):
        self.loglist[name] = log

    def delete_logfiles(self, requireConfirmation=True,
                        preserveCurrentLogs=True):
        logdir = self.logdir
        logfiles = glob.glob1(logdir, '*.log')
        if requireConfirmation:
            ans = raw_input('Delete all .log files in %s [y/n]? ' % logdir)
            if string.upper(ans[0]) != 'Y':
                return
        if preserveCurrentLogs:
            for lf in vgm.LogDispatcher.loglist.keys():
                logfiles.remove(lf)    
        for logfile in logfiles:
            os.remove(os.path.join(logdir, logfile))

