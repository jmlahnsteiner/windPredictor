# PWA Icons

This directory must contain two PNG icon files for the WindPredictor Progressive Web App:

| File | Size | Usage |
|------|------|-------|
| `icon-192.png` | 192×192 px | Android home screen, PWA manifest |
| `icon-512.png` | 512×512 px | Android splash screen, high-DPI displays |

Both files are also referenced by `<link rel="apple-touch-icon">` in the HTML head for iOS
"Add to Home Screen" support.

## Requirements

- **Format:** PNG with transparency (or solid background)
- **Shape:** Square — iOS and Android will mask to their own shapes (rounded rect / circle)
- **Content suggestion:** A wind/weather themed icon — e.g. a sailboat, compass rose, or wind
  arrow on the `#0077cc` theme-colour background

## How to add icons

1. Create or export your icon as a 512×512 PNG, save as `icons/icon-512.png`
2. Resize to 192×192, save as `icons/icon-192.png`
3. Commit both files to the repository — they will be picked up automatically by the
   GitHub Actions workflow (the Pages artifact is uploaded from the workspace root,
   which includes this `icons/` directory)

## Quick generation with ImageMagick (optional)

If you just want a placeholder to test with:

```bash
# Solid blue square with a white sailboat emoji (requires ImageMagick + a font that has ⛵)
convert -size 512x512 xc:#0077cc -fill white -font "Noto-Color-Emoji" \
  -pointsize 350 -gravity center -annotate 0 "⛵" icons/icon-512.png
convert icons/icon-512.png -resize 192x192 icons/icon-192.png
```
