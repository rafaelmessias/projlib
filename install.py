import sys
import subprocess
import ctypes


# https://gist.github.com/f0k/63a664160d016a491b2cbea15913d549
def check_for_cuda():
    libnames = ['libcuda.so', 'libcuda.dylib', 'cuda.dll']
    for libname in libnames:
        try:
            cuda = ctypes.CDLL(libname)
        except OSError:
            continue
        else:
            return True
    return False


if __name__ == '__main__':    
    dep = sys.argv[1]

    if dep == "ivis":
        cpu_or_gpu = 'gpu' if check_for_cuda() else 'cpu'
        module_name = f"ivis[{cpu_or_gpu}]"
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', module_name])
    
    else:
        print("Unknown dependency:", dep)
