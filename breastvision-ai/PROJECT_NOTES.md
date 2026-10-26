# Implementation notes

## What is included
- `prepare_data.py`: scans class directories, excludes masks, checks image readability, removes exact pixel duplicates, and creates a stratified manifest.
- `train.py`: trains EfficientNet-B0 with a frozen-backbone phase followed by full fine-tuning, class-weighted loss, AMP on CUDA, early stopping, and checkpoint saving.
- `evaluate.py`: reports held-out test metrics and creates a confusion matrix.
- `app.py`: Streamlit image upload and inference interface.
- `src/gradcam.py`: Grad-CAM visualization for the final feature block.

## Known limitations of this implementation
- Split is stratified by image label after exact-image deduplication. BUSI folder releases do not reliably expose patient IDs, so patient-level separation is not guaranteed.
- Exact pixel hashes do not detect visually similar images that differ in crop, contrast, compression, or overlay.
- Grad-CAM implementation is a qualitative overlay only.
- This code is not validated for medical use.
