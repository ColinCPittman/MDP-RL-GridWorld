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

    pyinstaller_cmd = [
        'pyinstaller',
        '--onefile',           
        '--windowed',          
        '--name', output_name,
        'src/gridworld_mdp.py'
    ]


    subprocess.check_call(pyinstaller_cmd)


    shutil.rmtree('build', ignore_errors=True)
    shutil.rmtree('__pycache__', ignore_errors=True)
    os.remove(f'{output_name}.spec')

    print(f"Successfully packaged {output_name}")

if __name__ == '__main__':
    package_application()