import sys
import traceback


class AITextException(Exception):
    def __init__(self, message):
        # Convert to string to avoid issues with non-string errors
        self.message = str(message)

        # Get current exception traceback
        exc_type, exc_value, exc_tb = sys.exc_info()

        if exc_tb:
            # exception occurred inside try/except
            tb = traceback.extract_tb(exc_tb)[-1]
            self.file = tb.filename
            self.line = tb.lineno
        else:
            # exception raised manually
            caller = traceback.extract_stack(limit=2)[0]
            self.file = caller.filename
            self.line = caller.lineno

        super().__init__(self.message)

    def __str__(self):
        return f"{self.message} (File: {self.file}, Line: {self.line})"


# if __name__=="__main__":
#     try:
#         logging.info("entered the function")
#         a=1/0
#     except Exception as e:
#         raise AITextException(e)
