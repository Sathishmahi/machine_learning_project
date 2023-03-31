import os
import sys


class CustomException(Exception):
    def __init__(self, error_msg: Exception, error_details: sys) -> None:
        # super().__init__(error_msg)
        self.error = CustomException.get_detailed_error_msg(
            error_msg=error_msg, error_details=error_details
        )

    @staticmethod
    def get_detailed_error_msg(error_msg: Exception, error_details: sys) -> str:
        """
        to extract the details from error_details obj

        Args:
            error_msg (Exception): Exception obj
            error_details (sys): sys obj

        Returns:
            str: error message
        """
        _, _, exc_tb = error_details.exc_info()
        except_line_number = exc_tb.tb_frame.f_lineno
        exact_lineno = exc_tb.tb_lineno
        file_name = exc_tb.tb_frame.f_code.co_filename
        error_message = f"""

        error occured the script [ {file_name} ] and 
        try-exception line number is [{except_line_number}] 
        exact line number is [{exact_lineno}] 
        error msg is [{error_msg}]
        
        """
        return error_message

    def __str__(self):
        return self.error
