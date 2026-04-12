"""Allow `python -m astrag` when the console script is not on PATH."""

from astrag.bench.cli import main

if __name__ == "__main__":
    main()
