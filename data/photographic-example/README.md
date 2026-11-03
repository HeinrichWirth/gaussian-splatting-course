# Структура в воде · Structure in water

Девять фотографий одной скульптуры сняты с разных сторон. Восемь кадров дают модели примеры для обучения. Кадр `view-04.jpg` остаётся для сравнения после обучения. Все девять положений камеры известны заранее. В папке `images/` лежат снимки, а в `sparse/0/` — параметры объектива, положения камер и 117 восстановленных точек в текстовом формате COLMAP.

Nine photographs show the same sculpture from different positions. Eight frames provide the training images. The ninth, `view-04.jpg`, is held out for evaluation. All camera poses are known. The JPEGs are in `images/`. Lens parameters, camera poses and 117 triangulated points are in `sparse/0/` in COLMAP's text format.

**Photographs and original camera calibration** © NVIDIA Corporation, 2026. The source is the Nikon sequence in `struktur28_auto` from [PhysicalAI-NuRec-PPISP](https://huggingface.co/datasets/nvidia/PhysicalAI-NuRec-PPISP). The subset, thumbnails and derived capture data are distributed under [Creative Commons Attribution 4.0 International](https://creativecommons.org/licenses/by/4.0/).

| Course image | Original file in `struktur28_auto/images/nikon/` | Split |
|---|---|---|
| view-00.jpg | DSC_4034.jpg | train |
| view-01.jpg | DSC_4037.jpg | train |
| view-02.jpg | DSC_4061.jpg | train |
| view-03.jpg | DSC_4058.jpg | train |
| view-04.jpg | DSC_4040.jpg | held out |
| view-05.jpg | DSC_4076.jpg | train |
| view-06.jpg | DSC_4043.jpg | train |
| view-07.jpg | DSC_4055.jpg | train |
| view-08.jpg | DSC_4046.jpg | train |

The original rectified images are 2000 × 1330 pixels. A central 1200 × 900 crop begins at (400, 215). Resampling this crop to 160 × 120 scales the focal lengths by 2/15. In COLMAP's coordinates measured from the image corner, the principal point moves from (1000, 665) to (80, 60). In the course renderer the first pixel center is (0, 0), so the principal point is (79.5, 59.5).

The world origin was shifted to the least-squares intersection of the eight training camera axes. World positions and camera baselines were then rescaled by the same factor to make the median camera distance 4 scene units. This similarity transform preserves pixel projections. World coordinates are in scene units.

For the sparse cloud, PyCOLMAP 4.2.1 extracted SIFT features from the eight training crops at 960 × 720, matched all training pairs and triangulated tracks using the supplied calibration. Stored points have at least two observations, a reprojection error at most 0.5 pixels at the published size, and a distance of at most 8 scene units from the origin. Each point color is the median of its training observations. The held-out RGB image enters only the evaluation. Camera poses come from NVIDIA's calibration of the full capture.

### Готовая сцена · Reference reconstruction

В уроке 20 можно открыть [сцену с 117 гауссианами](../../site/dist/assets/reconstruction/photo-reference.json). У каждого пятна есть центр в 3D, размеры, поворот, цвет и непрозрачность. Для обучения брались группы по 256 пикселей из восьми снимков. На снимке `view-04.jpg` при размере 80 × 60 ошибка MSE уменьшилась с 0.195159 до 0.009817. Силуэт виден, мелкие грани и отражения остаются размытыми.

The lesson 20 viewer opens the same 117-splat model. Each of its 1,200 updates sampled 256 pixels from a training camera. At 80 × 60 pixels the held-out view has a PSNR of 20.08 dB. For a closer look at the remaining blur, the viewer puts the target and the reconstruction side by side. The following command, run from the course root with its training dependencies, recreates the model in `static-scene-result/reference.json`.

```sh
python scripts/train_photo_reference.py
```
