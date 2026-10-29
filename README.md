# Gaussian Splatting Course

---

![Poster](assets/Poster.jpg)

---

**[🇷🇺 Русский](README_ru.md)**

## What This Is
Gaussian Splatting reconstructs a representation of a 3D scene from photographs and produces images from new viewpoints. Quality depends on the observations; surfaces absent from every input photo cannot be reliably recovered. The 2023 3DGS method enabled high-quality real-time rendering of such scenes on suitable hardware.

This course explains how it works. From the inside. From the math of a single element to running it on a GPU.

---

## Why I Made This

I started learning Gaussian Splatting and quickly realized: I can't freely read the papers. Too much math, too few examples. I need analogies, concrete numbers, code — not half-page formulas with no explanation of what they mean.

A free course that would take me from zero to understanding — I couldn't find one. In the paid segment, literally a couple of courses of questionable quality. So I started building my own. The kind that would make sense to me from day one. I dig into the topic, test it through examples, structure it, and write it up.

The course grows with me.

---

## Who This Is For

Engineers, enthusiasts, and anyone who writes code and wants to understand how this works.

I'm not a mathematician. I'm a CV/3D engineer.

---

## Prerequisites

- **Python** — familiarity with reading and changing a short script and working with functions, loops, and arrays.
- **PyTorch** — familiarity with tensors helps. We revisit the operations we need through examples and explain autograd and the training loop in the course.
- **Math** — ordinary arithmetic is enough to start. Vectors, matrices, derivatives, and other necessary concepts are introduced before use, with pictures and concrete numbers.

Before the first block, we set up the environment, run a ready-made example, and explore its controls. Each practical exercise will specify its dependencies, hardware requirements, and expected result.

---

## How Learning Works

**Visual result → hands-on experiment → concrete numbers → math step by step → code → understanding check.**

We first look at the result and try changing it. Then we investigate why it changes. The math explains behavior we have already observed.

Every formula we use is taken apart:

1. What problem it solves and where it comes from.
2. How to read every symbol; the inputs and outputs, their dimensions, and their valid ranges.
3. Why each operation is needed and the order in which it is performed.
4. A complete numerical calculation, without skipped intermediate steps.
5. What the result means on a graph, in an image, or in a scene.
6. How changing one parameter affects the result and what happens at boundary cases.
7. How each step maps to code.

New concepts — the exponential, variance, an inverse matrix, a derivative, a Jacobian — receive their own explanation before appearing in a larger formula. We check analogies against numbers and show their limits. Simplifications and approximations are identified explicitly.

Every lesson includes an interactive experiment or runnable example with adjustable parameters. In the closing exercise, we predict a change in the image, test the prediction, and explain the result.

---

## Available Materials

| Lesson | Reading on GitHub | Practice in Jupyter |
|---|---|---|
| 01 | [What Gaussian Splatting Is and What Problem It Solves](lessons/en/lesson-01-en.md) | [Scene exploration notebook](notebooks/en/lesson-01.ipynb) |
| 02 | [One Gaussian on a Line: Taking Apart the Bell Curve](lessons/en/lesson-02-en.md) | [Graphs and calculations notebook](notebooks/en/lesson-02.ipynb) |
| 03 | [A Gaussian on a Plane: From a Bell Curve to a Spot](lessons/en/lesson-03-en.md) | [Heatmap, hill, and slices of one spot](notebooks/en/lesson-03.ipynb) |
| 04 | [Shape and Rotation: Why We Need a Covariance Matrix](lessons/en/lesson-04-en.md) | [Ellipse rotation and matrix calculations](notebooks/en/lesson-04.ipynb) |
| 05 | [Several Spots: Color, Opacity, and Overlap](lessons/en/lesson-05-en.md) | [Color blending and a Gaussian editor](notebooks/en/lesson-05.ipynb) |

The notebooks contain the full lessons, code, figures, and interactive exercises. The reading versions preserve the text, code, and static results. GitHub shows the saved results; the interactive experiments run in Jupyter.

The first block takes us from one bell curve to a small image. The lesson 5 editor supports adding and removing spots, moving them, changing their sizes, rotation, color, and opacity, and reordering the layers. Scenes can be saved as JSON and loaded again: these parameters will be useful when we start adjusting them automatically in block 2.

### Running the Experiments

The examples run from a downloaded and extracted copy of the entire repository: the notebooks need the calculation and plotting modules in the course folder. All five lessons in the first block require only a CPU and Python 3.11 or newer; no GPU, PyTorch, or external-service accounts. Internet access is needed for the initial dependency installation and optional research links.

The commands run in a terminal opened in the course folder. The first creates a separate environment:

```text
python -m venv .venv
```

In Windows PowerShell, these commands install the dependencies in that environment and start JupyterLab:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m jupyterlab
```

On macOS or Linux, the equivalent commands are:

```sh
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m jupyterlab
```

With an existing Anaconda setup, the dependencies can be installed in the selected environment and Jupyter launched from there. The server and Python kernel use the environment containing those dependencies.

The first lesson is at `notebooks/en/lesson-01.ipynb`. In the open notebook, **Kernel → Restart Kernel and Run All Cells** restarts the kernel and runs every cell in order. Once execution finishes, the figures and parameter controls are ready. The scene example updates when a slider is released. The reset button restores the initial settings.

Text such as `VBox(...)` instead of sliders usually means widget support has not loaded. In that case, we check the `ipywidgets` installation in the kernel environment and restart Jupyter. **Shift+Enter** runs one cell and moves to the next.

Lesson text is edited in the notebooks. Running `python scripts/build_lessons.py` executes all ten notebooks in the first block and updates the Markdown versions and figures. The `--lessons 3 4 5` option limits the build to those lessons. The `--languages ru` and `--languages en` options select one language.

---

## Curriculum

Six blocks, 30 short lessons, and three continuing projects: **image reconstruction → a static 3D scene → a dynamic scene**. The final block addresses their size, speed, and compression.

This is the course development plan; currently available texts are listed above. Each block builds on the results of the previous ones.

### Block 1. How Gaussians Form an Image

We explore one Gaussian, learn to control its shape, and build an image from several soft spots.

| # | Lesson | Experiment and visible result |
|---|---|---|
| 01 | What Gaussian Splatting Is and What Problem It Solves | We explore a ready-made scene: centers, individual Gaussians, and the final image. We compare it with the source photographs and investigate its limits, including surfaces absent from every photo. |
| 02 | One Gaussian on a Line: Taking Apart the Bell Curve | We move the center and change the width. We build the formula from subtraction, division, squaring, and the exponential, connecting each operation to the graph. |
| 03 | A Gaussian on a Plane: From a Bell Curve to a Spot | We view one function as a heatmap, a hill, and equal-value contours, then turn a circle into an ellipse. |
| 04 | Shape and Rotation: Why We Need a Covariance Matrix | We rotate an ellipse and follow the matrix entries. We explore scale, variance, covariance, and the inverse matrix, then calculate values at selected points. |
| 05 | Several Spots: Color, Opacity, and Overlap | We build a simple image, change the order of two colored spots, and calculate a selected pixel's color by hand. |

**Result:** a small editor for Gaussians on a plane. Every parameter has a clear connection to a visible change.

### Block 2. How Gaussians Learn

We gradually replace manual adjustments with automatic updates and reconstruct a chosen image.

| # | Lesson | Experiment and visible result |
|---|---|---|
| 06 | Measuring Image Error | We compare a target with the result and build an error map, then calculate simple loss functions using a few pixels. |
| 07 | Derivatives and Gradients: Which Way to Move a Parameter | We fit one spot manually, plot the error, and connect its slope to the direction of an update. We also test the effect of step size. |
| 08 | Automatic Differentiation and the First Training Loop | We follow “parameter → image → loss → gradient → update,” work through the chain rule with numbers, and compare a numerical gradient check with PyTorch. |
| 09 | Reconstructing an Image with Gaussians | We train tens and hundreds of spots, compare initial positions, sizes, and element counts, and observe where detail is missing. |

**Result:** a program gradually reconstructs a chosen image, and every training step can be explained.

This teaching project is called **image reconstruction with Gaussians**. It is not an implementation of the separate 2D Gaussian Splatting method for reconstructing surfaces in 3D.

### Block 3. How a 3D Scene Becomes an Image

We add space and a camera, separating object motion, camera motion, and changes in the image.

| # | Lesson | Experiment and visible result |
|---|---|---|
| 10 | Coordinates and Transformations in 3D | We move a point and an object and rotate a coordinate frame. We explore vectors, matrix multiplication, and conversions between world and camera coordinates. |
| 11 | How a Camera and Perspective Work | We change the camera position and focal length, calculate pixel coordinates for a few points, and explain why distant objects look smaller. |
| 12 | A Gaussian in 3D: Shape, Scale, and Rotation | We turn a sphere into an ellipsoid and explore rotation matrices, axis and angle, then quaternions, their formulas, and normalization. We also check shape parameter constraints. |
| 13 | Projecting a Gaussian: From a Volume to a Spot | We view an ellipsoid from different directions and project its center and shape. Local changes explain the Jacobian and help test the limits of the approximation. |
| 14 | Rendering a Scene: Depth, Visibility, and Blending | We build a scene with several Gaussians, observe incorrect overlap ordering, and trace how the final frame is produced. |
| 15 | Why Color Changes with Viewing Direction | We compare constant color with view-dependent color, gradually introduce spherical harmonics, and show the contribution of individual functions and coefficients. |

**Result:** a teaching 3D renderer with a movable camera and controls for inspecting individual Gaussians.

### Block 4. Complete Static Gaussian Splatting

We combine training and rendering, then move from controlled examples to photographs we capture ourselves.

| # | Lesson | Experiment and visible result |
|---|---|---|
| 16 | Training a 3D Scene from Multiple Images | We start with a simple scene and known cameras, then test why matching one image is insufficient and how additional views constrain the solution. |
| 17 | How the Model Changes the Number of Gaussians | We explore cloning, splitting, pruning, and opacity resets, separate these operations from gradient updates, and observe changes in quality and memory use. |
| 18 | The Original Paper and the Fast Implementation | We map familiar components to Kerbl et al., explore screen tiles, culling, and GPU processing, and compare the teaching and fast implementations. |
| 19 | Preparing Data: Capture, Cameras, and COLMAP | We go from photographs to camera parameters and an initial point cloud, investigating poor coverage, motion blur, and camera reconstruction errors. |
| 20 | A First Captured Scene | We prepare a small dataset, train a model, save the result, and open it in a viewer. |
| 21 | Evaluating Quality and Diagnosing Artifacts | We test held-out views and investigate floaters, blur, flickering, and color errors. We explain metrics through images and numbers and measure speed and memory use. |

**Result:** a captured static scene and an understanding of how to distinguish data, model, and rendering problems. This gives us the tools to work independently with static GS.

### Block 5. Dynamic Scenes: Adding Time

We start with one moving element, then explore different ways to represent dynamics.

| # | Lesson | Experiment and visible result |
|---|---|---|
| 22 | What Changes When a Scene Moves | We compare camera and object motion, then explore data from one camera and multiple synchronized cameras, occlusions, and missing observations across space and time. |
| 23 | One Gaussian over Time | We define a trajectory, rotation, and changing shape, learn to recover simple motion, and compare interpolation with prediction beyond the observations. |
| 24 | Persistent Gaussians: Dynamic 3D Gaussians | We follow the same elements across frames and explore Luiten et al., motion constraints, and consistency between neighboring parts through a small example. |
| 25 | A Canonical Scene and Its Deformation: 4D-GS | We show how a shared canonical scene becomes the state at a selected time, then explore Wu et al.: representing space and time and predicting deformations. |
| 26 | Building a Simple Dynamic Model | We implement a selected approach on a small scene and test novel views, intermediate times, and motion stability. |

**Result:** a small dynamic scene with independent controls for time and camera position.

### Block 6. Size, Speed, and Compression

We use familiar scenes and measure the gains and losses from each change.

| # | Lesson | Experiment and visible result |
|---|---|---|
| 27 | Where File Size and Memory Use Come From | We calculate the cost of position, shape, color, and opacity, distinguishing saved model size, viewing memory, and training memory. |
| 28 | Compressing a Static Scene | We test removing low-value elements, reducing numerical precision, and simplifying color representation, then compare images, size, and speed before and after. |
| 29 | Compressing Dynamics | We investigate repetition across time and ways to store changes, comparing motion quality, size, and decoding time. |
| 30 | Final Experiment: Choosing a Method for a Task | We prepare several versions of one scene, compare quality, size, speed, and hardware requirements, justify the choice, and save the settings to reproduce the experiment. |

**Result:** a scene with measured characteristics and a reproducible optimization experiment.

### Supplementary Materials

Outside the core route: the ecosystem and current papers, surface and mesh reconstruction, the 2DGS method, and advanced GPU programming. These topics are added when a particular task calls for them.

---

## How to Go Through This

The lessons follow a sequence, with examples to run along the way. Each block builds on the previous one and ends with a working result.

Understanding a formula starts with its meaning. If a symbol or operation is unclear, we return to concrete numbers and the image, work through the missing step, and then continue.

We change one parameter at a time at first: we predict what will happen, test it, and explain any difference from our expectations. Then we combine changes. Intermediate results give us a record for comparing quality, speed, and size.

---

## Author

**Heinrich Wirth**

[@wirth_heinrich](https://x.com/wirth_heinrich)

---

## License

MIT.
