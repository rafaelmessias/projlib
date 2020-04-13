import sys
import subprocess
import ctypes


# https://gist.github.com/f0k/63a664160d016a491b2cbea15913d549
def check_for_cuda():
    for libname in ['libcuda.so', 'libcuda.dylib', 'cuda.dll']:
        try:
            ctypes.CDLL(libname)
        except OSError:
            continue
        else:
            return True
    return False


def pip_install(module_name):
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', module_name])

if __name__ == '__main__':    
    dep = sys.argv[1]

    if dep == "ivis":
        cpu_or_gpu = 'gpu' if check_for_cuda() else 'cpu'
        pip_install(f"ivis[{cpu_or_gpu}]")        

    if dep == 'umap':
        pip_install('umap-learn')
    
    else:
        print("Unknown dependency:", dep)
