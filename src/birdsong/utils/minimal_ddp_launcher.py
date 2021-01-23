import subprocess
import os, sys

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

    if len(sys.argv) < 2:
        print("Usage: {minimal_within_two_gpus_ddp.py | minimal_across_two_gpus_ddp.py}")
        sys.exit(1) 
    curr_dir = os.path.dirname(__file__)
    script_path = os.path.join(curr_dir, sys.argv[1])
    
    launcher = MinimalDDPLauncher()
    launcher.run_demo(script_path, 2)
