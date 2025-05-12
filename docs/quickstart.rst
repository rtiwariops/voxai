Quickstart
==========

Prerequisites
-------------

- **Python ≥3.7**  
- **Node.js ≥14**  
- **Electron** (global)  

  .. code-block:: bash

     npm install -g electron

Install & Run
-------------

1. Install from PyPI:

   .. code-block:: bash

      pip install voxai

2. Configure environment:

   .. code-block:: bash

      cp .env.example .env

   Edit `.env`:

   .. code-block:: ini

      GENAI_API_KEY=sk-…
      GENAI_MODEL=gemini-1.5-flash

3. Launch the app:

   .. code-block:: bash

      voxai

   On first launch VoxAI will automatically run ``npm install`` inside its bundled ``electron/`` folder, then open a desktop window.

