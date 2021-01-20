import subprocess
import os

class MinimalDDPLauncher:
   
    def run_demo(self, demo_script, world_size, goal):
        procs = []
        for rank in range(world_size):
            print(f"Starting {demo_script}[{rank}] of {world_size}")
            procs.append(subprocess.Popen([demo_script, 
                                           f"--rank={rank}"), 
                                           f"--goal={goal}"),
                                           ]))
            
        for proc in procs:
            proc.wait()

# ------------------------ Main ------------
if __name__ == '__main__':

    parser = argparse.ArgumentParser(prog=os.path.basename(sys.argv[0]),
                                     formatter_class=argparse.RawTextHelpFormatter,
                                     description="Test model parameter values or process drift"
                                     )

    parser.add_argument('-g', '--goal', choices=['parameters', 'drift'])
    
    args = parser.parse_args();

    curr_dir = os.path.dirname(__file__)
    script_path = os.path.join(curr_dir, 'minimal_ddp.py')
    
    launcher = MinimalDDPLauncher()
    world_size = 2
    launcher.run_demo(script_path, world_size, goal=args.goal)
