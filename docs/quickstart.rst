Quickstart
==========

Prerequisites
-------------

- **Python ≥3.7**  
- **Node.js ≥14**  
- **Electron** (global)  

  .. code-block:: bash

     npm install -g electron

**That's it!** VoxAI automatically installs Electron dependencies on first run.

**Important:** VoxAI captures system audio only (meetings, browser, apps) - no external microphone.

System Audio Setup
------------------

VoxAI captures **computer audio only** (no microphone) - perfect for meeting recordings, browser audio, and app sounds without room noise.

macOS Setup
^^^^^^^^^^^

.. code-block:: bash

   # 1. Install BlackHole
   brew install blackhole-2ch

   # 2. Install Ladiocast
   # Download from: https://existential.audio/ladiocast/

**Configure Ladiocast:**

1. **Input 1**: Set to your audio source (Built-in Input or system audio)
2. **Main**: Route Input 1 to Main (for speakers)
3. **Aux 1**: Set to BlackHole 2ch
4. **Enable**: Route Input 1 to Aux 1

**System Settings:**

- **Input**: BlackHole 2ch (VoxAI reads from here)
- **Output**: Built-in Output (you hear from here)

Windows (Built-in)
^^^^^^^^^^^^^^^^^^

WASAPI loopback support is built into Windows 10+ - VoxAI will auto-detect.

Linux (Built-in)
^^^^^^^^^^^^^^^^

PulseAudio monitor devices are auto-detected - no setup needed.

**Result:** You hear audio through speakers + VoxAI captures clean computer audio.

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

3. (Optional) **Customize the System Prompt**  
   By default, VoxAI uses a built-in generic, professional prompt. To override it, create a file named ``system_prompt.txt`` in the **current working directory** (the folder from which you run the command). For example:

   ::

     ~/projects/my_voxai/
     ├── .env
     ├── system_prompt.txt    ← Your custom prompt
     └── run_voxai.py         ← Or just “voxai” if installed globally

4. Launch the app:

   .. code-block:: bash

      voxai

   On first launch VoxAI will automatically run ``npm install`` inside its bundled ``electron/`` folder, then open a desktop window.

