import sys
import traceback
from logger import logging
class AITextException(Exception):
    def __init__(self, message: str):
        super().__init__(message)
        self.message = message

        # Get traceback of current exception
        _, _, exc_tb = sys.exc_info()

        if exc_tb:
            last = traceback.extract_tb(exc_tb)[-1]
            self.file = last.filename
            self.line = last.lineno
        else:
            # If raised manually
            caller = traceback.extract_stack()[-2]
            self.file = caller.filename
            self.line = caller.lineno
        logging.error(f"{self.message} | File: {self.file} | Line: {self.line}",exc_info=True)   

    def __str__(self):
        return f"{self.message} (File: {self.file}, Line: {self.line})"
    


if __name__=="__main__":
    try:
        logging.info("entered the function")
        a=1/0
    except Exception as e:
        raise AITextException(e)

