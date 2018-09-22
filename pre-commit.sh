#!/usr/bin/env bash


# First command gets names of files that have been Added, Copied, Modified
# Second command filters only Python files
# Put those file names in a file
#   (can't use xargs because we want to look at Pylint's exit code)
git diff --cached --name-only --diff-filter=ACM | grep ".*\.py$" > python_files_to_lint
if [ ! -s python_files_to_lint ]; then
    # No Python files being committed
    rm python_files_to_lint
    exit 0
fi
echo "Black\n"
black --line-length 100 --py36 python_files_to_lint
echo "flake8\n"
flake8 --max-line-length=100 python_files_to_lint
echo "MyPy\n"
mypy --strict python_files_to_lint

pylint_args="$(paste -s -d ' ' python_files_to_lint)"
pylint $pylint_args
exit_status=$?
rm python_files_to_lint
# See https://pylint.readthedocs.io/en/latest/user_guide/run.html#exit-codes
if [ $exit_status -eq 0 ]; then
    echo "No Pylint messages. Good job :)"
    exit 0
fi
if [ $exit_status -eq 32 ]; then
    echo "Pylint crashed!"
    exit 2
fi
if [ $(( $exit_status & 7 )) -gt 0 ]; then
    echo "Pylint found fatal/error/warning messages."
    exit 1
fi
echo "Ignoring Pylint refactor/convention messages."

exit 0
