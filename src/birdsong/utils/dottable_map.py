'''
Created on Dec 19, 2020

@author: Based on Stackoverflow answer by epool:
https://stackoverflow.com/questions/2352181/how-to-use-a-dot-to-access-members-of-dictionary

Extends dictionaries so that dot notation can be
used.
'''
import copy

class DottableMap(dict):
    """
    Flattens dicts; adds entries to resulting single dict with
    the kwargs.
    
    Example:
    m = DottableMap({'first_name': 'Eduardo'}, last_name='Pool', age=24, sports=['Soccer'])
    """
    def __init__(self, *args, **kwargs):
        super(DottableMap, self).__init__(*args, **kwargs)
        for arg in args:
            if isinstance(arg, dict):
                for k, v in arg.items():
                    self[k] = v

        if kwargs:
            for k, v in kwargs.items():
                self[k] = v

    def __getattr__(self, attr):
        return self.get(attr)

    def __setattr__(self, key, value):
        self.__setitem__(key, value)

    def __setitem__(self, key, value):
        super(DottableMap, self).__setitem__(key, value)
        self.__dict__.update({key: value})

    def __delattr__(self, item):
        self.__delitem__(item)

    def __delitem__(self, key):
        super(DottableMap, self).__delitem__(key)
        del self.__dict__[key]
        
    def __copy__(self):
        new_dottable_map = DottableMap()
        for key, val in self.items():
            new_dottable_map[key] = copy.copy(val)

    def getint(self, key, default=None):
        try:
            return int(self[key])
        except KeyError:
            return default
        except ValueError as e:
            raise TypeError(f"Value {self[key]} cannot be converted to an integer") from e
    
    def getfloat(self, key, default=None):
        try:
            return float(self[key])
        except KeyError:
            return default
        except ValueError as e:
            raise TypeError(f"Value {self[key]} cannot be converted to a float") from e

    def getboolean(self, key, default=None):
        try:
            the_str = self[key]
            return the_str in ('1', 'y', 'Y', 'yes', 'Yes', 'True')
        except KeyError:
            return default
        except ValueError as e:
            raise TypeError(f"Value {self[key]} cannot be converted to a bool") from e

    
