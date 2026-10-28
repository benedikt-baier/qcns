"""Build the QMath array bridge with the existing GNU C++ toolchain.

Equivalent CMake target: native/qmath. This helper also supports systems with
GCC/OpenMP installed but no CMake executable. No downloads or installations.
"""
import argparse
from pathlib import Path
import shutil
import subprocess
import sys


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--compiler', default='g++')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    compiler = shutil.which(args.compiler)
    if compiler is None:
        parser.error('GNU C++ compiler not found; supply --compiler with its path')
    root = Path(__file__).resolve().parents[1]
    native = root / 'native' / 'qmath'
    qmath = root / 'qcnspp' / 'QMath' / 'qmath'
    suffix = '.dll' if sys.platform == 'win32' else '.dylib' if sys.platform == 'darwin' else '.so'
    output = args.output or native / 'build' / ('qcns_qmath' + suffix)
    output.parent.mkdir(parents=True, exist_ok=True)
    sources = ['matrix_storage.cpp', 'dynamics/single_transformation.cpp',
               'dynamics/controlled_transformation.cpp', 'dynamics/cnot.cpp',
               'channels/dephasing.cpp', 'channels/amplitude_damping.cpp',
               'channels/depolarization.cpp', 'quantum_info/reductions.cpp']
    command = [compiler, '-std=c++20', '-O2', '-DNDEBUG', '-fopenmp', '-shared']
    if sys.platform == 'win32':
        command += ['-static-libgcc', '-static-libstdc++']
    else:
        command += ['-fPIC']
    command += ['-I', str(qmath / 'include'), str(native / 'bridge.cpp')]
    command += [str(qmath / 'src' / source) for source in sources]
    command += ['-o', str(output)]
    subprocess.run(command, check=True)
    if sys.platform == 'win32':
        # Keep the matching MinGW runtime beside the bridge for Python's secure
        # DLL search (which intentionally does not use PATH on recent Python).
        for name in ('libgomp-1.dll', 'libwinpthread-1.dll', 'libgcc_s_seh-1.dll'):
            runtime = Path(compiler).parent / name
            if runtime.is_file() and runtime.resolve() != (output.parent / name).resolve():
                shutil.copy2(runtime, output.parent / name)
    print(output)


if __name__ == '__main__':
    main()
