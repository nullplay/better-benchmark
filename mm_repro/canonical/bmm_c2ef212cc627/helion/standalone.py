"""Code generation failed. See codegen_status.json; do not benchmark this file."""
COMPILE_ERROR = 'NotImplementedError: Unsupported Helion preview op: check_bounds'

def call(args):
    raise RuntimeError(COMPILE_ERROR)

if __name__ == "__main__":
    raise RuntimeError(COMPILE_ERROR)
