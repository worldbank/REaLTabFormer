"""Entry point when invoked with python -m realtabformer."""  # pragma: no cover

if __name__ == "__main__":  # pragma: no cover
    import sys

    import realtabformer

    if "--version" in sys.argv:
        print(f"REaLTabFormer version: {realtabformer.__version__}")
