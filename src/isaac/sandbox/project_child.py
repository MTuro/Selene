"""Trusted bootstrap mounted into the isolated project sandbox, run with python -I."""

import ctypes
import errno
import json
import os
import resource
import sys


def main():
    # All hard limits are installed before loading any repository-controlled code.
    for kind, limit in [
        (resource.RLIMIT_CPU, 15),
        (resource.RLIMIT_AS, 512 << 20),
        (resource.RLIMIT_FSIZE, 1 << 20),
        (resource.RLIMIT_CORE, 0),
    ]:
        resource.setrlimit(kind, (limit, limit))

    lib = ctypes.CDLL("libseccomp.so.2")
    lib.seccomp_init.argtypes = [ctypes.c_uint32]
    lib.seccomp_init.restype = ctypes.c_void_p
    lib.seccomp_syscall_resolve_name.argtypes = [ctypes.c_char_p]
    lib.seccomp_rule_add.argtypes = [ctypes.c_void_p, ctypes.c_uint32, ctypes.c_int, ctypes.c_uint]
    lib.seccomp_load.argtypes = [ctypes.c_void_p]
    policy = lib.seccomp_init(0x00050000 | errno.EPERM)
    if not policy:
        sys.exit("Cannot initialize seccomp")
    for name in json.loads(sys.argv[1]):
        number = lib.seccomp_syscall_resolve_name(name.encode())
        if number >= 0 and lib.seccomp_rule_add(policy, 0x7FFF0000, number, 0):
            sys.exit("Cannot configure seccomp")
    if lib.seccomp_load(policy):
        sys.exit("Cannot enforce seccomp")
    os.execv(sys.argv[2], sys.argv[2:])


if __name__ == "__main__":
    main()
