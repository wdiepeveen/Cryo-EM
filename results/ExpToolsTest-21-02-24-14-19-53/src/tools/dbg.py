
import time
import logging

logger = logging.getLogger(__name__)

def dbg(level, message):

    global debug_level
    global debug_timer_start

    if debug_level in globals() or level <= debug_level:
        if debug_timer_start in globals():
            if level == 0:
                logger.info("{}".format(message))
            else:
                logger.info("{:9.4f} {}".format(time.time()-debug_timer_start,message))

def dbglevel(level):
    global debug_level
    debug_level = level


def dbglevel_atleast(level):
    global debug_level
    return debug_level in globals() or level <= debug_level

