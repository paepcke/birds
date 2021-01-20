import subprocess
import os

class MinimalDDPLauncher:
   
    def run_demo(self, demo_script, world_size):
        procs = []
        for rank in range(world_size):
            print(f"Starting {demo_script}[{rank}] of {world_size}")
            procs.append(subprocess.Popen([demo_script, str(rank), str(world_size)]))
            
        for proc in procs:
            proc.wait()

# ------------------------ Main ------------
if __name__ == '__main__':

    curr_dir = os.path.dirname(__file__)
    script_path = os.path.join(curr_dir, 'minimal_ddp.py')
    
    launcher = MinimalDDPLauncher()
    launcher.run_demo(script_path, 2)
