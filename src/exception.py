import sys 
# provides various functions and variables that are used to manipulate diff parts of the python runtime env
# any eception that is being contorlled, sys will have that info
import logging

def error_message_detail(error, error_detail:sys):
    # exc_info will give you details on which file, line the error has occured
    _,_,exc_tb = error_detail.exc_info() 
    file_name = exc_tb.tb_frame.f_code.co_filename
    error_message = "Error Occured in python script name [{0}] line number [{1}] error message [{2}]".format(
        file_name, exc_tb.tb_lineno, str(error)
    )
    return error_message


class CustomException(Exception):
    def __init__(self,error_message, error_detail:sys):
        super().__init__(error_message)
        self.error_message = error_message_detail(error_message, error_detail=error_detail )

    def __str__(self):
        return self.error_message
    

