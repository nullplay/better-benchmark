"""Code generation failed. See codegen_status.json; do not benchmark this file."""
COMPILE_ERROR = 'NotImplementedError: Cannot recover Helion tensor indices: 262144*((helion_y//128)) + (ModularIndexing(helion_y, 1, 128)) + 128*(ModularIndexing(helion_x + 2048*helion_y, 1, 2048))'

def call(args):
    raise RuntimeError(COMPILE_ERROR)

if __name__ == "__main__":
    raise RuntimeError(COMPILE_ERROR)
