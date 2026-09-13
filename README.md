# video-transfer — face swap build

Swaps the hero's face in `inputs/source.mp4` with the face in `inputs/face.jpg`
using GitHub Actions. Nothing runs on your computer.

## How to run

1. Commit these files to the repo (drag-and-drop in the GitHub web UI works;
   keep the folder structure, including `.github/workflows/faceswap.yml`).
2. Open the **Actions** tab → **Face swap video** → **Run workflow**.
   Leave the defaults and press the green **Run workflow** button.
3. Wait ~10–20 minutes (CPU runner). The job log shows frame progress.
4. Download the result from either:
   - the run page → **Artifacts** → `faceswap-result` (zip with `result.mp4` + `preview.jpg`), or
   - **Releases** → `Face swap run N` → `result.mp4`.

## To swap a different video or photo

Replace `inputs/source.mp4` and/or `inputs/face.jpg` (or upload under new
names and type those paths in the **Run workflow** form).

## Notes

- The swapper follows the largest face in frame and then tracks it between
  frames, so background passengers keep their own faces. Set `all_faces` to
  `true` in the form if you want every face replaced.
- Output keeps the original resolution, frame rate and audio.
- `inswapper_128.onnx` (InsightFace) is licensed for non-commercial use only.
  If the default download URL stops working, add a repository variable
  named `INSWAPPER_URL` (Settings → Secrets and variables → Actions → Variables)
  pointing to another copy of the file.
