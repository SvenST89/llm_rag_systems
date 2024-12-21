import os
import sys
import time
import datetime
import logging
import functools

def log_function(func):
    """
    A decorator that logs the function name, arguments, return value and internal logging statements.
    It measures the time taken for executing a function.
    You can use this function if you want to track functions and timing individually.
    To avoid errors of mixing up loggers, e.g. nothing is printed into the file or console, check this: https://stackoverflow.com/questions/55320048/python-logger-confusion
    
    The problem is resolved once you remove "__name__" from the instantiated logger, because then the logger is instantiated independently of the file.
    Otherwise if you set ".getLogger(__name__)", the logger will listen to the name "utils" as in this file and so will not print to console or file if
    you use the logger decorator in another file.
    """
    # os.makedirs(MODEL_DIR, exist_ok=True)
    LOG_DIR = os.path.join(os.getcwd(), "logs")
    #print(LOG_DIR)
    os.makedirs(LOG_DIR, exist_ok=True)
    # Configure the logging
    # the general log-level logs anything that is above the given level, e.g., INFO logs Warnings, Errors and Critical messages, but not DEBUG.
    # If we do not set a log-level at a specific Handler, this handler inherits the level from the general logger-level.
    logger = logging.getLogger()
    # set general log-level
    logger.setLevel("INFO")
    # Time stamp for log-file to avoid getting overwritten
    time_now = datetime.datetime.now()
    time_str = datetime.datetime.strftime(time_now, '%Y%m%d_%H%M%S')
    formatter = logging.Formatter('{asctime} - {levelname} - {lineno} - {message}', style="{", datefmt="%Y-%m-%d %H:%M:%S")
 
    # FILE HANDLER
    fileHandler = logging.FileHandler(filename=fr"{LOG_DIR}/{time_str}_{func.__name__}.log", mode="a", encoding="utf-8")
    fileHandler.setLevel("INFO")
    
    # Check if the handler already exists
    if not any(isinstance(handler, logging.FileHandler) and handler.baseFilename == fileHandler.baseFilename for handler in logger.handlers):
        logger.addHandler(fileHandler)
    
    # Set formatter for every handler
    for handler in logger.handlers:
        handler.setFormatter(formatter)
 
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # Log function call details
        logger.info(f"Calling function: {func.__name__}")
        #logger.info(f"Arguments: args [{args}] | kwargs [{kwargs}]")
 
        try:
            # add timing calculation
            start = time.time()
            # Call the original function
            result = func(*args, **kwargs)
            end = time.time()
            # Log the function´s return value
            logger.info(f"FUNCTION '{func.__name__}' RETURNED:\n{result}")
            # in minutes
            time_taken = round(float((end - start)/60), 2)
            logger.info(f"TIME TAKEN to execute function {func.__name__}: {time_taken} minutes")
        except Exception as e:
            # Log the exception if one is raised
            logger.error(f"Function {func.__name__} raised an exception:{e}")
            #time_taken = None
            # re-raise the exception after logging it
            raise
 
        return result
 
    return wrapper
 
 
# Trace function to capture calls within the function
def trace_calls(frame, event, arg):
    """
    The sys.settrace() function used in the decorator function below activates Python´s internal tracing system. Once this function is activated,
    Python will call the trace_calls() function every time an 'even' occurs, such as a function call.
    Each time a function is called during code execution (a 'call' event), Python passes the current stack 'frame' and the event type to this function'trace_calls'.
    
    Parameters / Help
    -----------------
        frame:
            A frame object from the internal Python interpreter which represents the current stack frame that contains infos such as 
            code object being executed (function name, file name, etc.), local variables, current execution line
        event:
            A string indicating the type of event that occurs, e.g., 'call' = function call, 'return' = function return, 'exception', 'line' = interpreter has reached new line.
        arg:
            This holds information that depends on the event type! For 'call', arg is None; for 'return' arg is the return value of the function; 
            for 'exception' arg is a tuple containing exception type, value and traceback! So you could access this inside this function here to log the args.
    """
    if event == 'call':
        code = frame.f_code
        func_name = code.co_name
        func_filename = code.co_filename
        func_lineno = frame.f_lineno
       
        # Here you could add some things that should be logged additionally!
   
    return trace_calls
 
# Decorator for logging function entry, exit, and internal calls
def logging_decorator(func):
    """
    This decorator logs all logging statements inside the decorated function and any log statements inside other functions that are called within the decorated function.
    It will write a log-file which has the name of the decorated function, but contains any log statements of all functions called inside this decorated function.
    In addition a console handler is added to check the execution inside the console. This handler inherits the Log-Level from the basicConfig level!
    
    Parameters / Help
    -----------------
        func:
            Decorated function.
    """
 
    # os.makedirs(MODEL_DIR, exist_ok=True)
    LOG_DIR = os.path.join(os.getcwd(), "logs")
    #print(LOG_DIR)
    os.makedirs(LOG_DIR, exist_ok=True)
    # Configure the logging
    # the general log-level logs anything that is above the given level, e.g., INFO logs Warnings, Errors and Critical messages, but not DEBUG.
    # If we do not set a log-level at a specific Handler, this handler inherits the level from the general logger-level.
    logger = logging.getLogger()
    # set general log-level
    logger.setLevel("INFO")
    # Time stamp for log-file to avoid getting overwritten
    time_now = datetime.datetime.now()
    time_str = datetime.datetime.strftime(time_now, '%Y%m%d_%H%M%S')
    formatter = logging.Formatter('{asctime} - {levelname} - {lineno} - {message}', style="{", datefmt="%Y-%m-%d %H:%M:%S")
 
    # FILE HANDLER
    fileHandler = logging.FileHandler(filename=fr"{LOG_DIR}/{time_str}_{func.__name__}.log", mode="a", encoding="utf-8")
    fileHandler.setLevel("INFO")
    
    # Check if the handler already exists
    if not any(isinstance(handler, logging.FileHandler) and handler.baseFilename == fileHandler.baseFilename for handler in logger.handlers):
        logger.addHandler(fileHandler)
    
    # Set formatter for every handler
    for handler in logger.handlers:
        handler.setFormatter(formatter)
 
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        logger.info(f"Entering {func.__name__} with arguments: {args} {kwargs}")
       
        # Enable the tracing of function calls
        sys.settrace(trace_calls)
       
        try:
            # add timing calculation
            start = time.time()
            result = func(*args, **kwargs)
            end = time.time()
            logger.info(f"{func.__name__} returned {result}")
            # in minutes
            time_taken = round(float((end - start)/60), 2)
            logger.info(f"Time taken to execute function {func.__name__}: {time_taken} minutes")
            return result
        except Exception as e:
            # Disable the trace after function execution
            #sys.settrace(None)
            # Log the exception if one is raised
            logger.error(f"Function {func.__name__} raised an exception:{e}")
            #time_taken = None
            # re-raise the exception after logging it
            raise
   
    return wrapper