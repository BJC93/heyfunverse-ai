# Optional face models

Both models are optional. Without them, face refinement uses OpenCV's built-in Haar cascade and
identity checks (Phase 2) are unavailable. Nothing is downloaded unless you run
`python ai-engine/fetch_face_models.py --yes`.

| Model | File | Size (Hugging Face listing) | Licence | Used for |
| --- | --- | --- | --- | --- |
| YuNet 2023mar | `face_detection_yunet_2023mar.onnx` | about 233 kB | MIT | finding faces |
| SFace 2021dec | `face_recognition_sface_2021dec.onnx` | 38.7 MB (int8 variant 9.9 MB) | Apache 2.0 (stated in the model README) | same-person similarity score |

Source: `huggingface.co/opencv/opencv_zoo`, folders `models/face_detection_yunet` and
`models/face_recognition_sface`. Files go to `storage/models/face/` (git-ignored).

## What was verified, and what was not

Verified from the published pages: file names, listed sizes, licence statements, the OpenCV
tutorial's use of `FaceDetectorYN` and `FaceRecognizerSF`, and that YuNet 2026may has the same
weights as 2023mar with dynamic input (MIT, aimed at OpenCV 5).

Not verified (the build sandbox could not reach the download host):
- the actual downloaded byte sizes and SHA-256 hashes (the fetch script records them, and refuses a
  file whose size is far from the listing);
- the minimum OpenCV version for 2023mar on your machine (the pages do not state one);
- the memory the models add at runtime. Measure it with
  `python ai-engine/check_env.py --probe-face`, which loads both models and prints the RAM growth.
  Expect small numbers for YuNet and tens of MB for SFace, but treat that as an estimate until measured.

The SFace licence covers the converted model file; the dataset it was trained on is not described
on the page I read, so check it yourself before any commercial use.

## Not approved, so not included

IP-Adapter, LoRA files and large upscalers are not downloaded or referenced by any script.
