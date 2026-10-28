"""Code generation failed. See codegen_status.json; do not benchmark this file."""
COMPILE_ERROR = 'NotImplementedError: Cannot recover Helion tensor indices: 49152*((helion_y//128)) + (ModularIndexing(helion_y, 1, 128)) + 128*(ModularIndexing(helion_x + 384*helion_y, 1, 384))'

def call(args):
    raise RuntimeError(COMPILE_ERROR)

if __name__ == "__main__":
    raise RuntimeError(COMPILE_ERROR)
