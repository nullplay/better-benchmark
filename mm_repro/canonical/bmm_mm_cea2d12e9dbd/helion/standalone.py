"""Code generation failed. See codegen_status.json; do not benchmark this file."""
COMPILE_ERROR = 'NotImplementedError: Cannot recover Helion tensor indices: 524288*((helion_y//1024)) + (ModularIndexing(helion_y, 1, 1024)) + 1024*(ModularIndexing(helion_x + 512*helion_y, 1, 512))'

def call(args):
    raise RuntimeError(COMPILE_ERROR)

if __name__ == "__main__":
    raise RuntimeError(COMPILE_ERROR)
