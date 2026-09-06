# SatQuery AI — Open Source Assets & Tool Registry

## 1. Vision-Language Models (Core & Reference)
* **Qwen3-VL-8B-Instruct** (`Qwen/Qwen3-VL-8B-Instruct` on Hugging Face): The primary VLM we will fine-tune via ROCm LoRA. Selected for its native multi-image interface and normalized bounding box support.
* **GeoChat** (`mbzuai-oryx/GeoChat` on GitHub): A premier remote-sensing-specific vision-language model. We will use this repo as a reference for structuring VQA prompts and text-grounding formats.

## 2. Mandated Datasets (Adaptation & Evaluation)
* **BigEarthNet.txt** (`BIFOLD-BigEarthNetv2-0/BigEarthNet.txt` on Hugging Face, arXiv: 2603.29630): The mandatory primary dataset containing Sentinel-1 (SAR) and Sentinel-2 (Optical) paired imagery with 9.6 million rich text annotations.
* **VRSBench** (`xiang709/VRSBench` on Hugging Face, `lx709/VRSBench` on GitHub): Required for single-image captioning, grounding, and VQA. Contains Very High Resolution (VHR) data.
* **CDVQA** (`YZHJessica/CDVQA` on GitHub): Required for training and evaluating bi-temporal change-based VQA.
* **LEVIR-CD & OSCD**: Used for training the Siamese change detection models.

## 3. Specialized Computer Vision & Fusion Tools
* **DOFA (Dynamic One-For-All)** (`zhu-xlab/DOFA` on GitHub): Foundational model for cross-modal optical/SAR joint reasoning — within SatQuery AI it serves as our optical–SAR fusion backbone, via its wavelength-conditioned patch embedding that accepts arbitrary band sets. We will access it natively via TorchGeo (`torchgeo.models.dofa_base_patch16_224`).
* **Open-CD / UCD** (`likyoo/Open-CD`, `AI4RS/UCD` on GitHub): Unified frameworks for bi-temporal change detection. We will reference their implementations of Siamese architectures like ChangeFormer and TinyCD.

## 4. Geospatial Python Libraries
* **TorchGeo** (`microsoft/torchgeo` on GitHub): Critical for building remote sensing dataloaders and running DOFA.
* **Rasterio**: The primary engine for geospatial ingestion, affine transform handling, and spectral index rendering (NDVI, NDBI).
