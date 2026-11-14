# Assets for captions and music (needed from Phase 3)

## Telugu font: Noto Sans Telugu

- Licence: SIL Open Font License 1.1 (OFL-1.1), copyright The Noto Project Authors, per the
  Fontsource listing. OFL permits use in videos and redistribution of the font file with its licence.
- Published as a variable font (`NotoSansTelugu[wdth,wght].ttf`, weight axis 100 to 900).
- Place the file in `storage/assets/fonts/`. `check_env.py` looks for `NotoSansTelugu*` there.
- Not yet verified: how your FFmpeg/libass handles the variable font's weight axis. Phase 3 will
  test a real Telugu caption render; if bold looks wrong, use static Regular and Bold files instead.

## FFmpeg

- Telugu needs complex-script shaping, which in FFmpeg means libass with HarfBuzz.
- The gyan.dev "essentials" and "full" Windows builds list libass, libfreetype, libfribidi and
  libharfbuzz among their libraries (checked on the builds page).
- `check_env.py` reads `ffmpeg -buildconf` and reports whether your build has libass, HarfBuzz and
  libx264.

## Music

Provide your own licensed files in `storage/assets/music/` (Phase 3). Silent output is supported
as a first-class option, so music is never required.
