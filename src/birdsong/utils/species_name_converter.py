#!/usr/bin/env python3
'''
Created on Jun 7, 2021

@author: paepcke
'''
import argparse
import csv
from enum import Enum
import os
import sys

'''
Convert between 4-letter, 6-letter, and scientific
names of bird species. Works from command line and
as a library.

        cnv = SpeciesNameConverter()
        cnv['WCPA', DIRECTION.FOUR_SCI]
            ---> 'PIONUS SENILIS'
        cnv['Pionus senilis', DIRECTION.SCI_SIX]
            ---> 'PIOSEN'
        cnv['WCPA', DIRECTION.FOUR_SIX]
            ---> 'PIOSEN'

'''

class DIRECTION(Enum):
    '''
    Distinguish between converation
    directions: six-letter convention, 
    four-letter convention, and scientific
    name, or in reverse. The conversions are
    implemented as lookup functions.
    The following table shows the conversions 
    and responsible methods.

	            4-Letter            6-Letter              SCINAME
	
	4-Letter        X            cls.four_to_six       cls.four_to_sci
	6-Letter  cls.six_to_four         X                cls.six_to_sci
	SCINAME   cls.sci_to_four    cls.sci_to_six              X

    The values of this enumerations are indices into
    the above dict table:
    '''
    FOUR_SIX = (0,1)
    FOUR_SCI = (0,2)
    SIX_FOUR = (1,0)
    SIX_SCI  = (1,2)
    SCI_FOUR = (2,0)
    SCI_SIX  = (2,1)

class SpeciesNameConverter:
    '''
    Converts between the four-letter, six-letter, 
    and scientific names of bird species naming 
    conventions. See 
    https://www.birdpop.org/pages/birdSpeciesCodes.php
    and also http://www.rockinrs.com/Bird_Alpha_Codes_English.pdf

    Acts like a dict with special properties:
    
        o Keys are case insensitive
        o Scienfic names may be entered with spaces,
          or with underscores instead of spaces
        o Readonly
        o The dict methods take an addition 'direction'
          argument inside the square brackets:
          
             SpeciesNameConverter()['WCPA', DIRECTION.FOUR_SCI]
        
          returns the scientific name of WCPA. Whereas:

             SpeciesNameConverter()['WCPA', DIRECTION.FOUR_SIX]
             
          returns the 6-letter equivalent of WCPA.
    '''

    initialized = False

    #------------------------------------
    # Constructor
    #-------------------

    @classmethod
    def initialize(cls):
        '''
        Read the translation table from file,
        and save it in a class variable. Needs 
        to be called only once per session.
        
        Also adds translations that are not in 
        the table.
        '''
        
        # Already initialized:
        if cls.initialized:
            return

        cls.six_to_four_dict = {}
        cls.four_to_six_dict = {}

        # 4-letter to scientific name and back
        cls.four_to_sci_dict = {}
        cls.sci_to_four_dict = {}
        
        # The standard conversion table
        # from https://www.birdpop.org/pages/birdSpeciesCodes.php:
        convertion_tbl_fname = os.path.join(
            os.path.dirname(__file__), 
            'data/birdSpeciesLetterCodes.csv')
        
        if not os.path.exists(convertion_tbl_fname):
            raise FileNotFoundError(f"Cannot find conversion table mapping 4/6-letter species names")
            
        with open(convertion_tbl_fname, 'r') as fd:
            reader = csv.DictReader(fd)
            for full_dict in reader:
                cls.six_to_four_dict[full_dict['SPEC6']] = cls.canonicalize_nm(full_dict['SPEC'])
                cls.four_to_six_dict[full_dict['SPEC']] = cls.canonicalize_nm(full_dict['SPEC6'])

                cls.four_to_sci_dict[full_dict['SPEC']] = cls.canonicalize_nm(full_dict['SCINAME'])
                sciname = cls.canonicalize_nm(full_dict['SCINAME'])
                cls.sci_to_four_dict[sciname] = cls.canonicalize_nm(full_dict['SPEC'])

        # Add additionals:
        cls.six_to_four_dict['AMADEC'] = 'CHHU'
        cls.four_to_six_dict['CHHU']   = 'AMADEC'
        cls.four_to_sci_dict['CHHU']   = 'AMAZILIA DECORA'
        cls.sci_to_four_dict['AMAXILIA DECORA'] = ['CHHU']
        
        cls.six_to_four_dict['HYLDEC'] = 'LESG'
        cls.four_to_six_dict['LESG']   = 'HYLDEC'
        cls.four_to_sci_dict['LESG']   = 'AMAZILIA DECORA'
        cls.sci_to_four_dict['AMAZILIA DECORA'] = ['LESG']
        
        # See comment for DIRECTION enumeration above
        # for the following dict lookup table for conversions.
        # The enumeration members such as FOUR_SIX are 
        # tuples that index into the following table. Thereby
        # the proper conversion dict can be retrieved:
        cls.conv_func_matrix = [
            [     None,         cls.four_to_six,     cls.four_to_sci],
            [cls.six_to_four,         None,          cls.six_to_sci ],
            [cls.sci_to_four,   cls.sci_to_six,          None       ],
            ]
        
        cls.initialized = True

    #------------------------------------
    # __getitem__
    #-------------------
    
    @classmethod
    def __getitem__(cls, nm_to_convert, *direction):
        '''
        Return conversion of item.

        Depending on the length of the arg,
        treat it as a 4-letter, 6-letter, or
        scientific name
        
        :param nm_to_convert: the item to convert
        :type nm_to_convert: str
        :param direction: which way to convert: FOUR_SIX,
            SIX_FOUR, etc.
        :type direction: DIRECTION
        :return: converted item
        :rtype: str
        :raise KeyError: if no conversion avaialable
        :raise TypeError: if direction arg is not one of the 
            DIRECTION members
        '''

        if not cls.initialized:
            cls.initialize()
            
        # Because __getitem__ is a magic method,
        # the args come as a tuple:
        
        nm_given  = nm_to_convert[0]
        direction = nm_to_convert[1]
        
        # Sanity checks:
        
        if type(direction) != DIRECTION:
            raise TypeError(f"The direction argument must be a DIRECTION member")

        if len(nm_given) == 4 and \
            direction not in [DIRECTION.FOUR_SCI,
                              DIRECTION.FOUR_SIX
                              ]:
            raise ValueError(f"For 4-letter codes, direction must be 6-letter or scienfic name")
        elif len(nm_given) == 6 and \
            direction not in [DIRECTION.SIX_FOUR,
                              DIRECTION.SIX_SCI
                              ]:
            raise ValueError(f"For 6-letter codes, direction must be 4-letter or scienfic name")
        elif len(nm_given) > 6 and \
            direction not in [DIRECTION.SCI_FOUR,
                              DIRECTION.SCI_SIX
                              ]:
            raise ValueError(f"For scientific names, direction must be 4-letter or 6-letter code")

        # Ensure proper format, such as
        # upper case, and no underscores
        # for names:
        
        nm = cls.canonicalize_nm(nm_given)
        row, col = direction.value
        conv_func = cls.conv_func_matrix[row][col]
        return conv_func(nm)

    #------------------------------------
    # four_to_six 
    #-------------------
    
    @classmethod
    def four_to_six(cls, nm):
        return cls.four_to_six_dict[nm]
    
    #------------------------------------
    # four_to_sci
    #-------------------
    
    @classmethod
    def four_to_sci(cls, nm):
        return cls.four_to_sci_dict[nm]
    
    #------------------------------------
    # six_to_four
    #-------------------
    
    @classmethod
    def six_to_four(cls,nm):
        return cls.six_to_four_dict[nm]

    #------------------------------------
    # six_to_sci
    #-------------------
    
    @classmethod
    def six_to_sci(cls,nm):
        return cls.four_to_sci_dict[cls.six_to_four_dict[nm]]

    #------------------------------------
    # sci_to_four
    #-------------------
    
    @classmethod
    def sci_to_four(cls,nm):
        return cls.sci_to_four_dict[nm]
    
    #------------------------------------
    # sci_to_six
    #-------------------
    
    @classmethod
    def sci_to_six(cls,nm):
        return cls.four_to_six_dict[cls.sci_to_four_dict[nm]]

    #------------------------------------
    # __len__ 
    #-------------------
    
    @classmethod
    def __len__(cls):
        
        if not cls.initialized:
            cls.initialize()

        # Dicts are all same len,
        # just use one of them:
        return len(cls.four_to_six_dict)
    
    #------------------------------------
    # keys 
    #-------------------
    
    @classmethod
    def keys(cls, direction):
        
        if not cls.initialized:
            cls.initialized
            
        if type(direction) != DIRECTION:
            raise TypeError(f"The direction argument must be a DIRECTION member")
        
        if direction in [DIRECTION.FOUR_SIX, DIRECTION.FOUR_SCI]:
            return cls.four_to_six_dict.keys()
            
        elif direction in [DIRECTION.SIX_FOUR, DIRECTION.SIX_SCI]:
            return cls.six_to_four_dict.keys()
            
        elif direction in [DIRECTION.SCI_FOUR, DIRECTION.SCI_SIX]:
            return cls.sci_to_four_dict.keys()
        
        else:
            raise ValueError(f"Argument {direction} not a member of DIRECTION enum.")


    #------------------------------------
    # values 
    #-------------------
    
    @classmethod
    def values(cls, direction):

        if type(direction) != DIRECTION:
            raise TypeError(f"The direction argument must be a DIRECTION member")


        # Values in dicts that are 4-letter?
        if direction in [DIRECTION.SIX_FOUR, DIRECTION.SCI_FOUR]:
            return cls.six_to_four_dict.values()
        
        # Values in dicts that are 6-letter?
        if direction in [DIRECTION.FOUR_SIX, DIRECTION.SCI_SIX]:
            return cls.four_to_six_dict.values()

        # Values in dicts that are sci names?
        if direction in [DIRECTION.FOUR_SCI, DIRECTION.SIX_SCI]:
            return cls.four_to_sci_dict.values()

        else:
            raise ValueError(f"Argument {direction} not a member of DIRECTION enum.")

    #------------------------------------
    # items 
    #-------------------
    
    @classmethod
    def items(cls, direction):
    
        if type(direction) != DIRECTION:
            raise TypeError(f"The direction argument must be a DIRECTION member")

        # Need to to get the precise dict
        if direction == DIRECTION.FOUR_SIX:
            return cls.four_to_six_dict.items()

        if direction == DIRECTION.FOUR_SCI:
            return cls.four_to_sci_dict.items()

        if direction == DIRECTION.SIX_FOUR:
            return cls.six_to_four_dict.items()
        
        if direction == DIRECTION.SIX_SCI:
            # No direct dict is maintained.
            items = zip(list(cls.six_to_four_dict.keys()),
                        list(cls.four_to_sci_dict.values())
                        )
            # Return a zip iterator:
            return items

        if direction == DIRECTION.SCI_FOUR:
            return cls.sci_to_four_dict.items()

        if direction == DIRECTION.SCI_SIX:
            # No direct dict is maintained.
            items = zip(list(cls.sci_to_four_dict.keys()),
                        list(cls.four_to_six_dict.values())
                        )
            # Return a zip iterator:
            return items

        else:
            raise ValueError(f"Argument {direction} not a member of DIRECTION enum.")

    #------------------------------------
    # __setitem__
    #-------------------

    @classmethod
    def __setitem__(cls, any_nm):
        raise NotImplementedError("cls is read-only")

    #------------------------------------
    # __delitem__
    #-------------------
    
    @classmethod
    def __delitem__(cls, any_nm):
        raise NotImplementedError("cls is read-only")

    #------------------------------------
    # canonicalize_nm
    #-------------------
    
    @classmethod
    def canonicalize_nm(cls, nm):
        '''
        Ensures that nm is of same form as 
        assumed in the internal dictionaries:
        all upper case, and no underscores: just
        spaces
        
        The nm argument may be a string, or a 
        tuple (name, DIRECTION)
        
        :param nm: item to canonicalize
        :type nm: str
        :return: the (possibly) modified input
        :rtype: str
        '''
        
        nm = nm.upper()
        nm = nm.replace('_', ' ')
        return nm

# ------------------------ Main ------------
if __name__ == '__main__':

    examples = '''
    Examples:
        species_name_converter.py WCPA four_sci
           ---> PIONUS SENILIS
        species_name_converter.py 'PIONUS SENILIS' sci_six
           ---> PIOSEN 
        species_name_converter.py 'Pionus senilis' sci_six
           ---> PIOSEN 
    '''
    parser = argparse.ArgumentParser(prog=os.path.basename(sys.argv[0]),
                                     formatter_class=argparse.RawTextHelpFormatter,
                                     description="Convert between-6, 4-letter, and scientific bird naming convention",
                                     epilog=examples
                                     )

    parser.add_argument('species',
                        help='Species as 4-letter, 6-letter, or scientific species name'
                        )

    parser.add_argument('direction',
                        choices=['four_six', 'four_sci',
                                 'six-four', 'six_sci',
                                 'sci_four', 'sci_six' 
                                 ],
                        help='Direction of conversion'
                        )

    args = parser.parse_args()
    
    # Converte the direction string
    # to a DIRECTION member:
    
    dir_str = args.direction
    if dir_str == 'four_six':
        dir = DIRECTION.FOUR_SIX
    elif dir_str == 'four_sci':
        dir = DIRECTION.FOUR_SCI
    elif dir_str == 'six_four':
        dir = DIRECTION.SIX_FOUR
    elif dir_str == 'six_sci':
        dir = DIRECTION.SIX_SCI
    elif dir_str == 'sci_four':
        dir = DIRECTION.SCI_FOUR
    elif dir_str == 'sci_six':
        dir = DIRECTION.SCI_SIX

    try:
        print(SpeciesNameConverter()[args.species, dir])
    except KeyError:
        print(f"No translation for {args.species}")

