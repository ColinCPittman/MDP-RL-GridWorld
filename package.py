import os
import platform
import shutil
import subprocess
import sys

def package_application():
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'pyinstaller'])

    system = platform.system().lower()
    arch = platform.machine().lower()

    output_name = f'gridworld-mdp-{system}-{arch}'
    python_dir = os.path.dirname(sys.executable)
    # Construct the path to pyinstaller.exe within that Python's Scripts dir
    pyinstaller_path = os.path.join(python_dir, 'Scripts', 'pyinstaller.exe')
    
    # Check if it exists before using it
    if not os.path.exists(pyinstaller_path):
        # Fallback or raise a more informative error
        pyinstaller_path = 'pyinstaller' # Try PATH as a fallback
        print(f"Warning: Could not find pyinstaller at {os.path.join(python_dir, 'Scripts')}, attempting to use PATH.")
    pyinstaller_cmd = [
        pyinstaller_path,
        '--onefile',           
        '--windowed',          
        '--name', output_name,
        'src/mdp_rl_gridworld.py'
    ]


    subprocess.check_call(pyinstaller_cmd)


    shutil.rmtree('build', ignore_errors=True)
    shutil.rmtree('__pycache__', ignore_errors=True)
    os.remove(f'{output_name}.spec')

    print(f"Successfully packaged {output_name}")

if __name__ == '__main__':
    package_application()