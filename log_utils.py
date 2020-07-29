'''
CUSTOM PRINT METHODS
'''
import colored
from colored import stylize
import sys

def bolden(msg, underline=False):
    bold = colored.attr("bold")# + colored.attr("underlined")
    return stylize(msg, bold)
    
def log(msg, bold=False):
    sys.stdout.write(bolden(msg) if bold else msg)
    sys.stdout.flush()
