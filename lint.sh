#!/bin/bash

flake8 . --count --max-complexity=10 --max-line-length=90 \
	--per-file-ignores="__init__.py:F401" \
	--exclude venv \
	--statistics
