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

    # I had some initial issues with pyinstaller, I added this because I thought it would force it to find the correct path more seamlessly. Leaving it in since it may help others
    pyinstaller_path = os.path.join(python_dir, 'Scripts', 'pyinstaller.exe')   
    if not os.path.exists(pyinstaller_path):
        pyinstaller_path = 'pyinstaller' 
        
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