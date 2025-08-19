# macOS Audio Setup for VoxAI

## Quick Setup

1. **Set BlackHole as Input Device:**
   - Open System Settings → Sound
   - Under "Input", select "BlackHole 16ch"

2. **Route Audio to BlackHole:**
   - Download Ladiocast (free): https://apps.apple.com/us/app/ladiocast/id411213048
   - Or use Audio MIDI Setup to create a Multi-Output Device

## Using Ladiocast (Recommended)

1. Open Ladiocast
2. In Input 1: Select "System Audio" or your current output
3. In Output Main: Select "BlackHole 16ch"
4. In Output Aux 1: Select "MacBook Air Speakers" (so you can hear)
5. Click the buttons to route Input 1 → Main and Input 1 → Aux 1

## Using Audio MIDI Setup (Alternative)

1. Open Audio MIDI Setup (in /Applications/Utilities)
2. Click "+" → "Create Multi-Output Device"
3. Check both:
   - MacBook Air Speakers
   - BlackHole 16ch
4. Set this Multi-Output Device as your system output

## Testing

1. Play any audio (YouTube, music, etc.)
2. Run VoxAI: `voxai`
3. Press spacebar to record while audio is playing
4. Release spacebar to transcribe

## Troubleshooting

- **No audio captured:** Make sure audio is actually playing through BlackHole
- **Can't hear audio:** Ensure you're routing to both BlackHole AND your speakers
- **Still not working:** Check the input device in System Settings is set to BlackHole