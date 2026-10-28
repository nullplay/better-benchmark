"""Code generation failed. See codegen_status.json; do not benchmark this file."""
COMPILE_ERROR = 'NotImplementedError: Cannot recover Helion tensor indices: helion_x + 512*helion_y + 262656*helion_z'

def call(args):
    raise RuntimeError(COMPILE_ERROR)

if __name__ == "__main__":
    raise RuntimeError(COMPILE_ERROR)
