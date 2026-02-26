<img width="1280" height="300" alt="VGGDrive" src="https://github.com/user-attachments/assets/9976a4f6-51d7-4d2d-aa35-1d9e46bde598" />

<h2 align="center">
✨ VGGDrive: Empowering Vision-Language Models ✨<br>
with Cross-View Geometric Grounding for Autonomous Driving
</h2>

## 📢 News
- **[2026/02/26]** 🚀 Released [VGGDrive NAVSIM v1 weights](#vggdrive-model-zoo) and inference code.
- **[2026/02/24]** 👉 We released our paper on [arXiv](https://arxiv.org/abs/2602.20794).
- **[2026/02/21]** 🎉🎉🎉 Accepted to CVPR 2026.


## 🔬 Project Overview

🧩 Conventional VLMs in autonomous driving “understand language but lack geometric insight.” Even when augmented with constructed Q&A data for auxiliary training, such approaches provide only superficial improvements and fail to address the core limitation in cross-view 3D spatial understanding.

💡 **VGGDrive** moves beyond data-level fixes and **charts a new course** by upgrading the capability structure itself. It introduces a mature 3D foundation model as a geometric backbone for VLMs, establishing a new technical paradigm that empowers Vision-Language Agents (VLAs) with 3D modeling capability and provides a scalable, sustainable pathway for enhancing autonomous driving systems.

<table>
<tr>
<td width="50%" valign="top">

<p style="text-align: justify;">
🛠️ The core innovation lies in the design of a <b>plug-and-play Cross-View Geometric Enhancer (CVGE)</b>. Through a hierarchical adaptive injection mechanism, VGGDrive achieves deep coupling between a frozen 3D foundation model and a VLM without altering the original VLM architecture. This mechanism efficiently injects 3D geometric features into the model, enabling genuine cross-view 3D geometric modeling capability for autonomous driving VLAs.
</p>

<p style="text-align: justify;">
📈 Importantly, VGGDrive is not limited to single-task optimization. It consistently improves performance across <b>five mainstream autonomous driving benchmarks</b>, covering cross-view risk perception, scene understanding, motion and state prediction, and trajectory planning, thereby enhancing the full pipeline from perception to decision-making.
</p>

</td>
<td width="50%" valign="top">

<img src="https://github.com/user-attachments/assets/9676c112-8140-4a12-aa02-5145f126d4a5" width="100%" />

</td>
</tr>
</table>

---
## 🏗️ Framework

<img width="3568" height="2208" alt="fig3_2" src="https://github.com/user-attachments/assets/ed54172b-0d78-49b6-940d-db1dea110700" />

<a name="vggdrive-model-zoo"></a>
## 🏛️ Model Zoo
| Model | Dataset | Download | Qwen_json |
|:-----:|:-------:|:--------:|:---------:|
| VGGDrive | NAVSIM | [ckpt](https://huggingface.co/wang-jie825/VGGDrive_model/tree/main) | [train & test](https://huggingface.co/datasets/wang-jie825/VGGDrive_Qwen_json/tree/main/navsim_cache) |
| VGGDrive | NuInstruct | | [train & test](https://huggingface.co/datasets/wang-jie825/VGGDrive_Qwen_json/tree/main/nuScenes_cache)  |
| VGGDrive | DriveLM | [submission.json](https://huggingface.co/datasets/wang-jie825/VGGDrive_Qwen_json/blob/main/DriveLM_submission.json) | [train & test](https://huggingface.co/datasets/wang-jie825/VGGDrive_Qwen_json/tree/main/nuScenes_cache)  |
| VGGDrive | OmniDrive | | [train & test](https://huggingface.co/datasets/wang-jie825/VGGDrive_Qwen_json/tree/main/nuScenes_cache)  |
| VGGDrive | NuScenes | | [train & test](https://huggingface.co/datasets/wang-jie825/VGGDrive_Qwen_json/tree/main/nuScenes_cache)  |
> ⚠️ **Prerequisite:**
> 
> Please download the pretrained VGGT model weights (`model.pt`) from [vggt](https://github.com/facebookresearch/vggt) and place it in the `./vggt` folder.
