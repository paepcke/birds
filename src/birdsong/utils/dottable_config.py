'''
Created on Dec 19, 2020

@author: paepcke
'''
from configparser import ConfigParser
import os, sys

from birdsong.utils.dottable_map import DottableMap

packet_root = os.path.abspath(__file__.split('/')[0])
sys.path.insert(0,packet_root)


class DottableConfigParser(DottableMap):
    '''
    Like the Python 3 configparser, with the
    following additions:
    
        o Dot notation. In addition to
         
            config['my_section]['my_option']
            
          can write:
          
            config.my_section.my_option
        
        o Additonal method getarray(section, option, default),
          which handles option values that are a comma separated
          list:
          
             my_option = /usr/lib/file.txt, [bluebell], 10
             
          with config.my_section.getarray(my_option)
          returning:
          
             ['/usr/lib/file.txt','[bluebell]','10']
              
        o Additional method getpathname(section, option, relative_to, default)
                  
          Resolves relative paths (those containing '.'/'..')
          in the given path.
             
             Examples:
                 os.path.abspath(os.path.join('/Users/doe/code', './src'))
                  --> '/Users/doe/code/src'
                 os.path.abspath(os.path.join('/Users/doe/code', './src/.'))
                  --> '/Users/doe/code/src'
                 os.path.abspath(os.path.join('/Users/doe/code', './src/..'))
                  --> '/Users/doe/code'
                 os.path.abspath(os.path.join('/Users/doe/code', './../src/..'))
                  --> '/Users/doe'
                 os.path.abspath(os.path.join('/Users/doe/code', '/Users/doe/code'))                  
                  --> '/Users/doe/code'
                  
        o Methods getint(), getboolean(), getfloat() and others, 
          which convert the strings in config options to the desired
          types.
            
    '''
    truth_values = ['1', 'yes', 'Yes', 'true', 'True', 'On', 'on']
    false_values = ['0', 'no', 'No', 'false', 'False', 'Off', 'off']
    legal_bools  = truth_values.copy()
    legal_bools.extend(false_values)
    
    #------------------------------------
    # __init__ 
    #-------------------

    def __init__(self, conf_src):
        '''
        Allow subclasses to pass either
        a path to a config file, or an
        already made config structure

        @param conf_src: file path to config file,
            or instance of ConfigParser
        @type conf_src: {src | ConfigParser}
        '''
        
        if type(conf_src) == str:
            built_in_config = ConfigParser()
            built_in_config.read(conf_src)
        else:
            built_in_config = conf_src
            
        for sec_name in built_in_config.sections():
            self[sec_name] = DottableMap(built_in_config[sec_name])
            
        self.sections = built_in_config.sections
            
    #------------------------------------
    # sections 
    #-------------------
    
    def sections(self):
        return self.sections

    #------------------------------------
    # getboolean 
    #-------------------
    
    def getboolean(self, section, option, default=None):
    
        option = str(option)
        
        try:
            val = self[section][option.lower()]
        except KeyError:
            # No such section and or option in config file
            if default is not None:
                val = default
            else:
                raise ValueError(f"Either section {section} or option {option} not in config file; nor default provided")

        if val is None and default is None:
            raise ValueError(f"Config section {section} option {option} are None, and no default provided")
        
        # If the default was given as 
        # True or False, just return that
        # value:
        if type(val) == bool:
            return val
        val_lower = val.lower()
        if val_lower in self.truth_values: 
            return True
        elif val_lower in self.false_values:
            return False
        else:
            raise ValueError(f"Boolean options must be one of \n    {self.legal_bools}\n   was '{val}'")

    #------------------------------------
    # getint 
    #-------------------
    
    def getint(self, section, option, default=None):
    
        try:
            val = self[section][option]
        except KeyError:
            # No such section and or option in config file
            if default is not None:
                val = default
            else:
                raise ValueError(f"Either section {section} or option {option} not in config file; nor default provided")
        return(int(val))
    
    #------------------------------------
    # getfloat 
    #-------------------
    
    def getfloat(self, section, option, default=None):

        try:
            val = self[section][option]
        except KeyError:
            # No such section and or option in config file
            if default is not None:
                val = default
            else:
                raise ValueError(f"Either section {section} or option {option} not in config file; nor default provided")

        return(float(val))

    #------------------------------------
    # getpath 
    #-------------------
    
    def getpath(self, section, option, relative_to=None, default=None):
        '''
        Access configuration section/option. The value is
        understood to be a path. If relative_to is a path,
        the relative_to path is path-joined with the
        config value
        
        @param section: config section
        @type section: str
        @param option: config option within the section 
        @type option: str
        @param relative_to: a path or None
        @type relative_to: {str | None}
        @param default: if section and/or option do not exist 
        @type default: {None | str}
        '''
        try:
            if relative_to is None:
                val = self[section][option]
            else:
                val = self.expand_path(self[section][option], relative_to)
        except KeyError:
            # No such section and or option in config file
            if default is not None:
                if relative_to is None:
                    val = default
                else:
                    val = self.expand_path(default, relative_to)
            else:
                raise ValueError(f"Either section {section} or option {option} not in config file; nor default provided")

        return(val)

    #------------------------------------
    # getarray
    #-------------------
    
    def getarray(self, section, option, default=None):
        '''
        Expect an option that is a comma separated list of
        items. 

               my_list = /foo/bar, 5, [abc]

        returns for the my_list option:

               ['/foo/bar', '5', '[abc]']


        @param section: config file section
        @type section: str
        @param option: option name
        @type option: str
        @param default: return value if option does not exist
        @type default: Any
        '''

        try:
            val = self[section][option]
        except KeyError:
            # No such section and or option in config file
            if default is not None:
                val = default
            else:
                raise ValueError(f"Either section {section} or option {option} not in config file; nor default provided")

        # Remove white space around it:
        val = val.strip()
        
        # Value will be a string either like:
        #
        #    '[/foo/bar.txt, 10, fum.txt]'
        # or '/foo/bar.txt, 10, fum.txt'
        #
        if val.startswith('[') and val[-1] == ']':
            # Safely evaluate the string to get
            # an array, without allowing the eval
            # any access to built-ins or other functions:
            #arr = eval(val,
            #           {"__builtins__":None},    # No built-ins at all
            #           {}                        # No additional func
            #           )
            raise ValueError(f"List-like options must be simple comma-separated strings without brackets: {val}")
        
        str_arr = val.split(',')
        # Clean up spaces after commas in the option str:
        str_arr = [el.strip() for el in str_arr]
        
        return(str_arr)

    # -------------------- Utils -------------
    
    #------------------------------------
    # expand_path 
    #-------------------

    def expand_path(self, dotted_path, relative_to_path):
        '''
        Resolves relative paths (those containing '.'/'..')
        in the given path.
        
        Examples:
            os.path.abspath(os.path.join('/Users/doe/code', './src'))
             --> '/Users/doe/code/src'
            os.path.abspath(os.path.join('/Users/doe/code', './src/.'))
             --> '/Users/doe/code/src'
            os.path.abspath(os.path.join('/Users/doe/code', './src/..'))
             --> '/Users/doe/code'
            os.path.abspath(os.path.join('/Users/doe/code', './../src/..'))
             --> '/Users/doe'
        
        @param dotted_path: path to expand, such as ../src
        @type path: str
        @param relative_to_path: path to which dotted_path
            is to be resolved
        @type relative_to_path: str
        @return absolute, combined path
        @rtype: str
        '''
        
        expanded_path = os.path.abspath(os.path.join(relative_to_path, dotted_path))
        return expanded_path