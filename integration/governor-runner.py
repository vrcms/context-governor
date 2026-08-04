import sys, os, traceback

# pythonw.exe has no console — sys.stdout/stderr are None.
# uvicorn's logging formatter calls sys.stdout.isatty() which crashes on None.
devnull = open(os.devnull, 'w')
sys.stdout = devnull
sys.stderr = devnull

try:
    from contextmanager.launcher import main
    main()
except Exception:
    # next to this script, so the runner works from any checkout
    logpath = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           '_crash.log')
    with open(logpath, 'w') as f:
        traceback.print_exc(file=f)
    raise
