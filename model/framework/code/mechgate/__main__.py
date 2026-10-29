# -*- coding: utf-8 -*-
"""python -m mechgate is equivalent to python -m mechgate.cli"""
import sys
from .cli import main

if __name__ == "__main__":
    sys.exit(main())
