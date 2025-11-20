#!/bin/bash -e
# run_tests.sh - entrypoint for tests, fully venv-aware and auto-installs test tools

TEST=$1
set -o pipefail
cd "$(dirname "$0")"/..

# Ensure we're in a virtual environment
if [[ -z "$VIRTUAL_ENV" ]]; then
    echo "Activating virtual environment..."
    if [[ -f "venv/bin/activate" ]]; then
        source venv/bin/activate
    else
        echo "Virtual environment not found. Creating one..."
        python -m venv venv
        source venv/bin/activate
    fi
fi

# Use venv pip and python explicitly
PIP="python -m pip"
PYTHON="python"

function pinfo() { echo -e "\033[32m${1}\033[m" >&2; }
function pwarn() { echo -e "\033[33m${1}\033[m" >&2; }
function perr() { echo -e "\033[31m${1}\033[m" >&2; }

# Install flarefly into the venv
setup-flarefly() {
    pinfo "Installing flarefly in virtual environment..."
    $PIP install --upgrade --force-reinstall --no-deps -e .
}

# Install a test tool if not already installed
install-tool() {
    TOOL=$1
    if ! type $TOOL &>/dev/null; then
        pinfo "Installing $TOOL..."
        $PIP install $TOOL
    fi
}

test-pylint() {
    install-tool pylint
    pinfo "Running test: pylint"
    find flarefly -name '*.py' | xargs pylint
}

test-flake8() {
    install-tool flake8
    pinfo "Running test: flake8"
    flake8 flarefly --count --select=E9,F63,F7,F82 --show-source --statistics
    flake8 flarefly --count --exit-zero --max-complexity=10 --max-line-length=127 --statistics
 }

test-pytest() {
    setup-flarefly
    install-tool pytest
    pinfo "Running test: pytest"
    pytest tests/test_data_handler.py
    pytest tests/test_massfitter_binned.py
    pytest tests/test_massfitter_unbinned.py
}

test-all() {
    test-pylint
    test-flake8
    test-pytest
}

# Check parameters
[[ $# == 0 ]] && test-all
while [[ $# -gt 0 ]]; do
    case "$1" in
    all) test-all ;;
    pylint) test-pylint ;;
    flake8) test-flake8 ;;
    pytest) test-pytest ;;

    --quiet)
        function pinfo() { :; }
        function pwarn() { :; }
        ;;
    --help)
        pinfo "run_tests.sh: entrypoint to launch tests locally or on CI"
        pinfo ""
        pinfo "Normal usage:"
        pinfo "    run_tests.sh [parameters] [test|all]   # no arguments: test all!"
        pinfo ""
        pwarn "Specific tests:"
        pwarn "    run_tests.sh pylint                    # test with pylint"
        pwarn "    run_tests.sh flake8                    # test with flake8"
        pwarn "    run_tests.sh pytest                    # test with pytest"
        pwarn ""
        pwarn "Parameters:"
        pwarn "    --help                                 # print this help"
        pwarn "    --quiet                                # suppress messages (except errors)"
        exit 1
        ;;
    *)
        perr "Unknown option: $1"
        exit 2
        ;;
    esac
    shift
done
