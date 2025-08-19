# VoxAI Audio Setup for Calls/Meetings

## Your Requirements:
✅ Hear computer audio through speakers  
✅ VoxAI captures only computer audio (no microphone)  
✅ Others can hear your voice in calls  

## Setup with Ladiocast (Recommended)

### 1. Download Ladiocast
- Free from App Store: https://apps.apple.com/us/app/ladiocast/id411213048

### 2. Configure Ladiocast

**Input 1:** System Audio (or Built-in Output)
- This captures all computer sounds

**Output Routing:**
- Main: BlackHole 16ch (for VoxAI to capture)
- Aux 1: MacBook Air Speakers (so you can hear)
- Click both routing buttons for Input 1

### 3. System Settings

**Sound Output:** MacBook Air Speakers (or your headphones)
**Sound Input:** MacBook Air Microphone (for calls)

### 4. In Your Call/Meeting App

**Microphone:** MacBook Air Microphone (or external mic)
**Speaker:** System Default (your speakers/headphones)

## How It Works:

```
Computer Audio → Ladiocast → BlackHole (VoxAI captures this)
                          ↘→ Speakers (you hear this)

Your Voice → Microphone → Call/Meeting App → Other Person

VoxAI Never Hears: Your voice, room noise, or anything from microphone
```

## Testing Your Setup:

1. Open Ladiocast with above settings
2. Play a YouTube video - you should hear it
3. Run `voxai` and press spacebar while video plays
4. VoxAI should transcribe the video audio
5. Join a test call - others should hear your mic, not computer audio

## Important Notes:

- **VoxAI Only Captures:** Computer audio routed through BlackHole
- **Your Mic:** Completely separate, only goes to calls
- **No Feedback Loop:** Your voice never reaches VoxAI
- **Call Audio:** If you want VoxAI to transcribe call audio, the other person's voice will be captured (but not yours)

## Pro Tip for Interviews:

This setup is perfect for:
- Transcribing interview questions from video calls
- Capturing YouTube tutorials or courses
- Recording system audio without background noise
- Getting AI responses without others hearing them