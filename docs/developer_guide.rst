Developer Guide
===============

Install from source
-------------------

.. code-block:: bash

   git clone https://github.com/rtiwariops/voxai.git
   cd voxai
   pip install -e .
   npm install -g electron
   voxai

Releasing a New Version
-----------------------

We provide a helper script ``release.sh``:

.. code-block:: bash

   ./release.sh <new-version> "<commit-message>"

This will:

1. Bump `setup.py` to `<new-version>`  
2. Update `.gitignore` with build artifacts  
3. Commit & tag (`v<new-version>`)  
4. Push to GitHub  
5. Build and upload to PyPI  

