                 Scripts

split_files
-----------

     - Create directory ../Testing, and replicate the subdir
       tree there
     - Randomly select x% of files for each subdirectory,
       and move them to their respective ../Testing/... destination.

          data_root
                   Training
          Species1        Species2
              file1        file4
              file2        file5
              file3        file6

     Becomes:

          data_root
                   Training                        Testing
          Species1        Species2        Species1        Species2 
                           file4              file1        file6
              file2        file5             
              file3                     

     where the files in each subdir of Testing is x% of the corresponding
     files in the subdires under Training.

   USES random_file_selector.py
