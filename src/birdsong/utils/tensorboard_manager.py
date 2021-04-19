'''
Created on Feb 12, 2021

@author: paepcke
'''

import fnmatch
import shutil
import subprocess
import sys, os

class TensorBoardManager(object):
    '''
    Clients can start and stop the tensorboard
    service for any port/event-root-dir. 
    
    Instances of Chrome or Firefox can be started
    and stopped, and pointed to the tensorboard
    manager.
    '''
    tensorboard_server = None
    web_browser = None
    
    # Stay away from the standard
    # tensorboard port (6006) in case
    # client has a server running there:
    
    tensorboard_port_default = 6007
    
    def __init__(self,
                 logdir,
                 port=None, 
                 new_tensorboard_server=True):

        if port is None:
            port = self.tensorboard_port_default
            
        self.port = port
        self.curr_dir = os.path.dirname(__file__)
        
        if new_tensorboard_server or self.tensorboard_server is None:
            self.start_tensorboard_server(logdir, self.port)

        if self.web_browser is None:
            self.start_webbrowser(self.port)
            
    #------------------------------------
    # start_tensorboard_server 
    #-------------------
    
    def start_tensorboard_server(self, logdir=None, port=None):
        

        if logdir is None:
            logdir = os.path.join(self.curr_dir, 'runs')
        
        if port is None:
            port = self.tensorboard_port_default
            
        tb_exec = self.find_executable('tensorboard')
    
        if tb_exec is None:
            msg = f"Cannot find tensorboard; if running in IDE, set path by hand"
            raise FileNotFoundError(msg)
        
        self.tensorboard_server = subprocess.Popen([tb_exec, 
                                                    '--logdir', logdir, 
                                                    '--port', str(port),
                                                    '--reload_interval', '1' # update every sec
                                                    ],
                                                    stdout=subprocess.PIPE,
                                                    stderr= subprocess.PIPE
                                                    )
        return

    #------------------------------------
    # stop_tensorboard_server 
    #-------------------
    
    def stop_tensorboard_server(self):
        
        if self.tensorboard_server is not None:
            self.tensorboard_server.kill()
            self.tensorboard_server.wait(0.5)
            self.tensorboard_server = None

    #------------------------------------
    # start_webbrowser 
    #-------------------
    
    def start_webbrowser(self, port):
        
        if self.web_browser is not None:
            raise RuntimeError("The start_webbrowser() method must only be called once; close tensorboard_manager in between.")

        browser_exec_file = self.find_browser()
        
        self.web_browser = subprocess.Popen([browser_exec_file, 
                                             f"http://localhost:{port}"],
                                             )

    #------------------------------------
    # stop_webbrowser 
    #-------------------
    
    def stop_webbrowser(self):

        if self.web_browser is None:
            # We never started a browser
            return
        
        self.web_browser.kill()
        # Allow browser to die:
        self.web_browser.wait(0.5)
        self.web_browser = None

    #------------------------------------
    # find_browsers 
    #-------------------

    def find_browser(self):
        
        chrome_exec  = self.find_executable('chrome')
        if chrome_exec is not None:
            return chrome_exec
        
        firefox_exec = self.find_executable('firefox')
        if firefox_exec is not None:
            return firefox_exec
        
        safari_exec  = self.find_executable('safari')
        if safari_exec is not None:
            return safari_exec

        raise FileNotFoundError("Found none of chrome, firefox, or safari")

    #------------------------------------
    # close 
    #-------------------

    def close(self):
        '''
        Free resources. Stop web and tensorboard 
        servers if they are running
        '''
        self.stop_tensorboard_server()
        self.stop_webbrowser()

    #------------------------------------
    # find_tensorboard_executable 
    #-------------------

    def find_executable(self, application_name):
        '''
        Given an application name, such as 
        'chrome', 'firefox', 'safari', or
        'tensorboard, return the path to the
        respective executable. 
        
        :param application_name: name of application to find
        :type application_name: str
        :return: full path to application, or None if not found
        :rtype {str | None}
        '''
        
        if sys.platform == "linux" or sys.platform == "linux2":
            exec_path  = shutil.which(application_name)
            
        elif sys.platform == "darwin":
            # OS X 
            # See whether 'which' finds the application:

            while True:
                exec_path = shutil.which(application_name)
                if exec_path is not None:
                    break
                
                #********** HACK ************
                # When working within an IDE, environments
                # can sometimes cause find/which tools to
                # fail. And searching the /Applications folders
                # can be slow. So, check the special cases
                # chrome, firefox, and tensorboard...
                
                if application_name == 'chrome':
                    exec_path = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" 
                    if os.path.isfile(exec_path):
                        break
                if application_name == 'firefox':
                    exec_path = '/Applications/Firefox.app/Contents/MacOS/firefox'
                    if os.path.isfile(exec_path):
                        break
                if application_name == 'safari':
                    exec_path = '/Applications/Safari.app/Contents/MacOS/Safari'
                    if os.path.isfile(exec_path):
                        break

                if application_name == 'tensorboard':
                    exec_path = f"{os.getenv('HOME')}/anaconda3/envs/birds/bin/tensorboard"
                    if os.path.isfile(exec_path):
                        break
                    
                # Walk the /Applications dir,
                # which is slow:
                print("Searching Mac /Applications folder: sloooow...")
                exec_path = self.find_in_applications(application_name)
                print("Done searching Mac /Applications folder")
                break
    
            if exec_path is None:
                return None
            
        # Make sure what we found is executable:
        try:
            fmode = os.stat(exec_path)
        except FileNotFoundError:
            # Found *some* file name, but it doesn't exist:
            return None

        if fmode.st_mode & 0o00500 != 0o500:
            # Even owner does not have execute and read
            # permissions:
            exec_path = None
            # Ensure application permissions are
            # at least Owner execute/read:

        return exec_path

    #------------------------------------
    # find_in_applications 
    #-------------------
    
    def find_in_applications(self, application_name):
        '''
        Find application by name below the 
        /Applications directory on MacOS. Return
        the name of the executable. If none found,
        return None; if more than one found, raise
        ValueError 

        :param application_name: name of application executable to find
        :type application_name: str
        :return path to executable or None
        :rtype {None | str}
        :raise ValueError: if multiple executables are found 
        '''

        root_dir = '/Applications'
        file_list = []
         
        # Walk through directory
        for dir_name, _sub_dirs, file_list in os.walk(root_dir):
            for file_name in file_list:
                # Match the application name precisely:
                if fnmatch.fnmatch(file_name, application_name): 
                    file_list.append(os.path.join(dir_name, file_name))

        if len(file_list) > 1:
            raise ValueError(f"Multiple executables for {application_name} found")
        
        return file_list[0]

# -------------------- Utils ------------------


# ------------------------ Main ------------
if __name__ == '__main__':
    TensorBoardManager()
