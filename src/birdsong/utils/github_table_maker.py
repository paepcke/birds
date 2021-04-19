'''
Created on Mar 12, 2021

@author: paepcke
'''
import itertools


class GithubTableMaker:
    '''
    classdocs
    '''

    #------------------------------------
    # make_table
    #-------------------

    @classmethod
    def make_table(cls, content_dict):
        '''
        Create github markdown table.
        Content dict must be like:
        
           'col_header' : ['foo', 'bar'],
           'row_labels' : ['row1', 'row2', 'row3'],
           'rows'       ; [[1,2],
                           [3,4],
                           [5,6]
                           ]

        :param content_dict: content information
        :type content_dict: {str | [Any]}
        :return the markdown table text
        :rtype str
        '''
        
        col_header = content_dict['col_header']
        row_labels = content_dict['row_labels']
        have_row_labels = len(row_labels) > 0
        
        # Find the widest text of the table:
        flat_dict_vals = col_header.copy()
        flat_dict_vals.extend(content_dict['row_labels'])
        # The following obscurity flattens
        # nested lists:
        try:
            flat_dict_vals.extend(list(itertools.chain(*content_dict['rows'])))
        except TypeError:
            # content_dict['rows'] is not a nested
            # list; it's just a plain list:
            flat_dict_vals.extend(content_dict['rows'])
        flat_vals_lengths = [len(str(el)) for el in flat_dict_vals]
        max_width = max(flat_vals_lengths)

        # Plus 1 is for the row labels column
        num_cols = 1*have_row_labels + len(col_header)
        
        # Column width worth of dashes
        col_dashes = '-'*max_width
        # Column width worth of spaces 
        col_spaces = ' '*max_width 
        
        # List of col-width sized dashes:
        #    ['-------', '-------', '-------']
        sep_sections = [col_dashes]*num_cols
        # Sep line for between each row, with 
        # pipes in the right places:
        sep_line = f"|{'|'.join(sep_sections)}|\n"
        
        # Put space padding around each side
        # of each header:
        #    ['   col1   ', '   col2   ', ...]
        padded_col_header = [cls._pad_column(head, max_width)
                             for head in col_header
                             ]

        # Top (i.e. header) line. First a col's
        # worth of empty for the row-label 'column':
        #   "|          |   col1   |   col2   | ..."
        if have_row_labels:
            # If row lables, the header row needs
            # an empty col above the label 'col':
            tbl = f"|{col_spaces}|{'|'.join(padded_col_header)}|\n"
        else:
            tbl = f"|{'|'.join(padded_col_header)}|\n"
        tbl += sep_line
        
        # Add the rows:
        for row_num, row_arr in enumerate(content_dict['rows']):
            
            # For single-col tables, row_arr
            # will be a single number, causing
            # a type error in the padded_row 
            # list comprehension. So:
            if type(row_arr) != list:
                row_arr = [row_arr]
            
            if not have_row_labels:
                row_str = f"|"
            else:
                row_str = f"|{cls._pad_column(row_labels[row_num], max_width)}|" 
            padded_row = [cls._pad_column(cell_val, max_width)
                             for cell_val in row_arr
                             ]
            row_str += f"{'|'.join(padded_row)}|\n"
            tbl += row_str
            tbl += sep_line
        #print(tbl)
        return tbl
            

    #------------------------------------
    # pad_column
    #-------------------
    
    @classmethod
    def _pad_column(cls, content, col_width):

        # Ensure we have strings, not
        # other types like float that may
        # sit in table cells:
        content = str(content)
        
        needed_spaces = col_width - len(content)
        if needed_spaces < 0:
            raise ValueError(f"Column content {content} longer than col width ({col_width})")
        
        spaces_per_side = needed_spaces // 2
        # If needed spaces is odd, make the left
        # side padding a bit longer:
        spaces_left = spaces_per_side + needed_spaces % 2
        
        col_content = f"{' '*spaces_left}{content}{' '*spaces_per_side}"

        return col_content

# ------------------------ Main ------------
if __name__ == '__main__':
    
    content_with_labels =\
              {'col_header' : ['col1', 'looong col2', 'col3'],
               'row_labels' : ['rooooow1', 'row2'],
               'rows'       : [['Cell 0_1', 'cell 0_2', 'ceeeeeeeel 0_3'],
                               ['Cell 1_1', 'cell 1_2', 'ceeeeeeeel 1_3']
                               ]
               }

    content_no_labels = \
              {'col_header' : ['col1', 'looong col2', 'col3'],
               'row_labels' : [],
               'rows'       : [['Cell 0_1', 'cell 0_2', 'ceeeeeeeel 0_3'],
                               ['Cell 1_1', 'cell 1_2', 'ceeeeeeeel 1_3']
                               ]
               }

    tm = GithubTableMaker()
    
    # Test with row labels:
    tbl_with_labels = tm.make_table(content_with_labels)
    correct = '|              |     col1     |  looong col2 |     col3     |\n|--------------|--------------|--------------|--------------|\n|   rooooow1   |   Cell 0_1   |   cell 0_2   |ceeeeeeeel 0_3|\n|--------------|--------------|--------------|--------------|\n|     row2     |   Cell 1_1   |   cell 1_2   |ceeeeeeeel 1_3|\n|--------------|--------------|--------------|--------------|\n'
    assert(tbl_with_labels == correct)
    
    tbl_no_labels = tm.make_table(content_no_labels)
    correct = '|     col1     |  looong col2 |     col3     |\n|--------------|--------------|--------------|\n|   Cell 0_1   |   cell 0_2   |ceeeeeeeel 0_3|\n|--------------|--------------|--------------|\n|   Cell 1_1   |   cell 1_2   |ceeeeeeeel 1_3|\n|--------------|--------------|--------------|\n'
    assert(tbl_no_labels == correct)
    
    print("Passed test")
    
