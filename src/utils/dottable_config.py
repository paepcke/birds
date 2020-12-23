'''
Created on Dec 19, 2020

@author: paepcke
'''
from configparser import ConfigParser
from . dottable_map import DottableMap

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
              
          
            
            
    '''
    truth_values = ['1', 'yes', 'Yes', 'true', 'True', 'On', 'on']
    false_values = ['0', 'no', 'No', 'false', 'False', 'Off', 'off']
    legal_bools  = truth_values.copy()
    legal_bools.extend(false_values)
    
    #------------------------------------
    # __init__ 
    #-------------------

    def __init__(self, conf_path):
        '''
        Constructor
        '''
        self.config = ConfigParser()
        self.config.read(conf_path)
            
        for sec_name in self.config.sections():
            self[sec_name] = DottableMap(self.config[sec_name])
            
        self.sections = self.config.sections
            
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
