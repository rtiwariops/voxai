# Volume Control Setup for VoxAI

## The Issue
BlackHole is a virtual audio device that doesn't support volume adjustment. It always captures at 100% volume.

## Solutions

### Option 1: Control App Volume (Recommended)
- Adjust volume in the app you're using (YouTube, Spotify, etc.)
- This affects both what you hear AND what VoxAI captures

### Option 2: Use Audio Hijack (Advanced)
1. Download Audio Hijack from Rogue Amoeba
2. Create a session that:
   - Captures System Audio
   - Adds Volume control block
   - Outputs to both Speakers and BlackHole
3. Adjust volume in Audio Hijack

### Option 3: Use SoundSource
1. Download SoundSource (menu bar app)
2. Control individual app volumes
3. Route specific apps to BlackHole

### Quick Volume Tips:
- **Mute VoxAI Capture**: Temporarily switch input from Multi-Output to just MacBook Speakers
- **Lower Capture Volume**: Reduce volume in the source app (browser, media player)
- **Keyboard Shortcuts**: Use F10/F11/F12 for mute/volume control (affects speakers only)

## Current Setup Reminder:
```
Your Audio → Multi-Output Device → BlackHole (100% to VoxAI)
                                 → Speakers (Adjustable volume)
```

The volume keys on your keyboard only affect the speakers portion, not BlackHole.