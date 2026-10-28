"""Code generation failed. See codegen_status.json; do not benchmark this file."""
COMPILE_ERROR = 'NotImplementedError: Helion preview supports atomic_add mutations'

def call(args):
    raise RuntimeError(COMPILE_ERROR)

if __name__ == "__main__":
    raise RuntimeError(COMPILE_ERROR)
