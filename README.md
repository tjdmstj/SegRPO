# SegRPO Audio Samples

Audio samples for "SegRPO: Group Relative Policy Optimization with Segment-Level Advantages for Code-Switching TTS".

The demo page is `index.html`, served with GitHub Pages.

## Layout

```
audio/<language>/<system>/<utterance>.wav
```

Systems are `GT`, `SFT`, `DPO`, `CS3`, and `SegRPO`. To add an utterance, place one wav file per system in the matching folders and add an entry to `LANGUAGES` at the top of the script in `index.html`.
