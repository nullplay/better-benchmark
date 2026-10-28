"""Code generation failed. See codegen_status.json; do not benchmark this file."""
COMPILE_ERROR = 'NotImplementedError: Cannot recover Helion tensor indices: helion_y + 1184*((helion_z//6)) + 196*(ModularIndexing(helion_z, 1, 6))'

def call(args):
    raise RuntimeError(COMPILE_ERROR)

if __name__ == "__main__":
    raise RuntimeError(COMPILE_ERROR)
