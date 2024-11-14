import logging

from systemds.context import SystemDSContext

# Create a context and if necessary (no SystemDS py4j instance running)
# it starts a subprocess which does the execution in SystemDS
with SystemDSContext() as sds:
    # Full generates a matrix completely filled with one number.
    # Generate a 5x10 matrix filled with 4.2
    m = sds.full((5, 10), 4.20)
    # multiply with scalar. Nothing is executed yet!
    m_res = m * 3.1
    # Do the calculation in SystemDS by calling compute().
    # The returned value is an numpy array that can be directly printed.
    logging.info(m_res.compute())
    # context will automatically be closed and process stopped