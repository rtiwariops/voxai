Quickstart
==========

Prerequisites
-------------

- **Python ≥3.7**  
- **Node.js ≥14**  
- **Electron** (global)  

  .. code-block:: bash

     npm install -g electron

- **macOS Audio Loopback (BlackHole 2ch + Ladiocast)**:

  1. **Install BlackHole 2ch**::

       brew install blackhole-2ch

  2. **Create a Multi-Output Device**  
     - Open **Audio MIDI Setup**  
     - Click “+” → **Create Multi-Output Device**  
     - Check both **BlackHole 2ch** and **MacBook Pro Speakers**

  3. **Select Multi-Output Device**  
     System Preferences → Sound → Output → **Multi-Output Device**

  4. **Configure Ladiocast**  
     - **Input 1**: **BlackHole 2ch** → route to **Main**  
     - **Main Output**: **MacBook Pro Speakers**  
     - Mute other channels

  This will route all system and microphone audio into VoxAI.

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

