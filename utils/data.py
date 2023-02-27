import re

# Access dictionary with dot notation
class DotDict(dict):    
    """dot.notation access to dictionary attributes"""      
    def __getattr__(*args):        
        val = dict.get(*args)         
        return DotDict(val) if type(val) is dict else val     
    __setattr__ = dict.__setitem__     
    __delattr__ = dict.__delitem__

# Alphanumeric sorting
def alpha_sort(iterable): 
    """ Sort the given iterable in the way that humans expect.""" 
    if iterable[0] and not isinstance(iterable[0], str): return sorted(iterable)
    else:
        convert = lambda text: int(text) if text.isdigit() else text 
        alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
        return sorted(iterable, key = alphanum_key)