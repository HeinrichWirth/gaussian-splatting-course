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

I'm not a mathematician. I'm a CV/3D engineer. If I figured it out — so can you.

---

## Prerequisites

- **Python** — be comfortable reading and changing a short script and working with functions, loops, and arrays.
- **PyTorch** — familiarity with tensors helps. We revisit the operations we need through examples and explain autograd and the training loop in the course.
- **Math** — ordinary arithmetic is enough to start. Vectors, matrices, derivatives, and other necessary concepts are introduced before use, with pictures and concrete numbers.

Before the first block: set up the environment, run a ready-made example, and learn the experiment controls. Each practical exercise will specify its dependencies, hardware requirements, and expected result.

---

## How Learning Works

**Visual result → hands-on experiment → concrete numbers → math step by step → code → understanding check.**

First, look at the result and try changing it. Then investigate why it changes. The math explains behavior you have already observed.

Every formula we use is taken apart:

1. What problem it solves and where it comes from.
2. How to read every symbol; the inputs and outputs, their dimensions, and their valid ranges.
3. Why each operation is needed and the order in which it is performed.
4. A complete numerical calculation, without skipped intermediate steps.
5. What the result means on a graph, in an image, or in a scene.
6. How changing one parameter affects the result and what happens at boundary cases.
7. How each step maps to code.

New concepts — the exponential, variance, an inverse matrix, a derivative, a Jacobian — receive their own explanation before appearing in a larger formula. We check analogies against numbers and show their limits. Simplifications and approximations are identified explicitly.

Every lesson includes an interactive experiment or runnable example with adjustable parameters. It ends with an exercise: predict a change in the image, test the prediction, and explain the result.

---

## Available Materials

| Lesson | Read on GitHub | Practice in Jupyter |
|---|---|---|
| 01 | [What Gaussian Splatting Is and What Problem It Solves](lessons/en/lesson-01-en.md) | [Scene exploration notebook](notebooks/en/lesson-01.ipynb) |
| 02 | [One Gaussian on a Line: Taking Apart the Bell Curve](lessons/en/lesson-02-en.md) | [Graphs and calculations notebook](notebooks/en/lesson-02.ipynb) |
| 03 | [A Gaussian on a Plane: From a Bell Curve to a Spot](lessons/en/lesson-03-en.md) | [Heatmap, hill, and slices of one spot](notebooks/en/lesson-03.ipynb) |
| 04 | [Shape and Rotation: Why We Need a Covariance Matrix](lessons/en/lesson-04-en.md) | [Rotate an ellipse and work through its matrix](notebooks/en/lesson-04.ipynb) |
| 05 | [Several Spots: Color, Opacity, and Overlap](lessons/en/lesson-05-en.md) | [Color blending and a Gaussian editor](notebooks/en/lesson-05.ipynb) |

The notebooks contain the full lessons, code, figures, and interactive exercises. The reading versions preserve the text, code, and static results. GitHub does not run the sliders; open the notebook in Jupyter for the experiments.

The first block takes us from one bell curve to a small image. In the lesson 5 editor, you can add and remove spots, move them, change their sizes, rotation, color, and opacity, and reorder the layers. Save a scene as JSON and load it again: these parameters will be useful when we start adjusting them automatically in block 2.

### Running the Experiments

Download and extract the entire repository: the notebooks need the calculation and plotting modules in the course folder. All five lessons in the first block require only a CPU and Python 3.11 or newer; no GPU, PyTorch, or external-service accounts. Internet access is needed for the initial dependency installation and optional research links.

Open a terminal in the course folder. Create a separate environment:

```text
python -m venv .venv
```

In Windows PowerShell:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m jupyterlab
```

On macOS or Linux:

```sh
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m jupyterlab
```

If you already use Jupyter through Anaconda, install the dependencies in your selected environment and launch Jupyter from it. The server and Python kernel must use an environment with the required dependencies.

In Jupyter, open `notebooks/en/lesson-01.ipynb`. Select **Kernel → Restart Kernel and Run All Cells**. Wait for execution to finish, then change the parameters. The heavier scene example updates when you release a slider. The reset button restores the initial settings.

If you see text such as `VBox(...)` instead of sliders, check that `ipywidgets` is installed in the kernel environment and restart Jupyter. You can also run cells individually with **Shift+Enter**.

Lesson text is edited in the notebooks. Running `python scripts/build_lessons.py` executes all ten notebooks in the first block and updates the Markdown versions and figures. To rebuild selected lessons, add, for example, `--lessons 3 4 5`. To build one language, add `--languages ru` or `--languages en`.

---

## Curriculum

Six blocks, 30 short lessons, and three continuing projects: **image reconstruction → a static 3D scene → a dynamic scene**. The final block addresses their size, speed, and compression.

This is the course development plan; currently available texts are listed above. Each block builds on the results of the previous ones.

### Block 1. How Gaussians Form an Image

Explore one Gaussian, learn to control its shape, and build an image from several soft spots.

| # | Lesson | Experiment and visible result |
|---|---|---|
| 01 | What Gaussian Splatting Is and What Problem It Solves | Explore a ready-made scene: centers, individual Gaussians, and the final image. Compare it with the source photographs and investigate its limits, including surfaces absent from every photo. |
| 02 | One Gaussian on a Line: Taking Apart the Bell Curve | Move the center and change the width. Build the formula from subtraction, division, squaring, and the exponential; connect each operation to the graph. |
| 03 | A Gaussian on a Plane: From a Bell Curve to a Spot | View one function as a heatmap, a hill, and equal-value contours. Turn a circle into an ellipse. |
| 04 | Shape and Rotation: Why We Need a Covariance Matrix | Rotate an ellipse and follow the matrix entries. Explore scale, variance, covariance, and the inverse matrix; calculate values at selected points. |
| 05 | Several Spots: Color, Opacity, and Overlap | Build a simple image. Change the order of two colored spots and calculate a selected pixel's color by hand. |

**Result:** a small editor for Gaussians on a plane. Every parameter has a clear connection to a visible change.

### Block 2. How Gaussians Learn

Gradually replace manual adjustments with automatic updates and reconstruct a chosen image.

| # | Lesson | Experiment and visible result |
|---|---|---|
| 06 | Measuring Image Error | Compare a target with the result and build an error map. Calculate simple loss functions using a few pixels. |
| 07 | Derivatives and Gradients: Which Way to Move a Parameter | Fit one spot manually, plot the error, and connect its slope to the direction of an update. Test the effect of step size. |
| 08 | Automatic Differentiation and the First Training Loop | Follow “parameter → image → loss → gradient → update.” Work through the chain rule with numbers and compare a numerical gradient check with PyTorch. |
| 09 | Reconstructing an Image with Gaussians | Train tens and hundreds of spots. Compare initial positions, sizes, and element counts; observe where detail is missing. |

**Result:** a program gradually reconstructs a chosen image, and every training step can be explained.

This teaching project is called **image reconstruction with Gaussians**. It is not an implementation of the separate 2D Gaussian Splatting method for reconstructing surfaces in 3D.

### Block 3. How a 3D Scene Becomes an Image

Add space and a camera. Distinguish object motion, camera motion, and changes in the image.

| # | Lesson | Experiment and visible result |
|---|---|---|
| 10 | Coordinates and Transformations in 3D | Move a point and an object, and rotate a coordinate frame. Explore vectors, matrix multiplication, and conversions between world and camera coordinates. |
| 11 | How a Camera and Perspective Work | Change the camera position and focal length. Calculate pixel coordinates for a few points and explain why distant objects look smaller. |
| 12 | A Gaussian in 3D: Shape, Scale, and Rotation | Turn a sphere into an ellipsoid. Explore rotation matrices, axis and angle, then quaternions, their formulas, and normalization; check shape parameter constraints. |
| 13 | Projecting a Gaussian: From a Volume to a Spot | View an ellipsoid from different directions. Project its center and shape; explain the Jacobian through local changes and test the limits of the approximation. |
| 14 | Rendering a Scene: Depth, Visibility, and Blending | Build a scene with several Gaussians. Observe incorrect overlap ordering and trace how the final frame is produced. |
| 15 | Why Color Changes with Viewing Direction | Compare constant color with view-dependent color. Introduce spherical harmonics gradually and show the contribution of individual functions and coefficients. |

**Result:** a teaching 3D renderer with a movable camera and controls for inspecting individual Gaussians.

### Block 4. Complete Static Gaussian Splatting

Combine training and rendering, then move from controlled examples to your own photographs.

| # | Lesson | Experiment and visible result |
|---|---|---|
| 16 | Training a 3D Scene from Multiple Images | Start with a simple scene and known cameras. Test why matching one image is insufficient and how additional views constrain the solution. |
| 17 | How the Model Changes the Number of Gaussians | Explore cloning, splitting, pruning, and opacity resets. Separate these operations from gradient updates; observe changes in quality and memory use. |
| 18 | The Original Paper and the Fast Implementation | Map familiar components to Kerbl et al. Explore screen tiles, culling, and GPU processing. Compare the teaching and fast implementations. |
| 19 | Preparing Data: Capture, Cameras, and COLMAP | Go from photographs to camera parameters and an initial point cloud. Investigate poor coverage, motion blur, and camera reconstruction errors. |
| 20 | Your First Captured Scene | Prepare a small dataset, train a model, save the result, and open it in a viewer. |
| 21 | Evaluating Quality and Diagnosing Artifacts | Test held-out views. Investigate floaters, blur, flickering, and color errors. Explain metrics through images and numbers; measure speed and memory use. |

**Result:** your own static scene and an understanding of how to distinguish data, model, and rendering problems. At this point, you can work independently with static GS.

### Block 5. Dynamic Scenes: Adding Time

Start with one moving element, then explore different ways to represent dynamics.

| # | Lesson | Experiment and visible result |
|---|---|---|
| 22 | What Changes When a Scene Moves | Compare camera and object motion. Explore data from one camera and multiple synchronized cameras, occlusions, and missing observations across space and time. |
| 23 | One Gaussian over Time | Define a trajectory, rotation, and changing shape. Learn to recover simple motion; compare interpolation with prediction beyond the observations. |
| 24 | Persistent Gaussians: Dynamic 3D Gaussians | Follow the same elements across frames. Explore Luiten et al., motion constraints, and consistency between neighboring parts through a small example. |
| 25 | A Canonical Scene and Its Deformation: 4D-GS | Show how a shared canonical scene becomes the state at a selected time. Explore Wu et al.: representing space and time and predicting deformations. |
| 26 | Your Own Simple Dynamic Model | Implement a selected approach on a small scene. Test novel views, intermediate times, and motion stability. |

**Result:** a small dynamic scene with independent controls for time and camera position.

### Block 6. Size, Speed, and Compression

Use familiar scenes and measure the gains and losses from each change.

| # | Lesson | Experiment and visible result |
|---|---|---|
| 27 | Where File Size and Memory Use Come From | Calculate the cost of position, shape, color, and opacity. Distinguish saved model size, viewing memory, and training memory. |
| 28 | Compressing a Static Scene | Test removing low-value elements, reducing numerical precision, and simplifying color representation. Compare images, size, and speed before and after. |
| 29 | Compressing Dynamics | Investigate repetition across time and ways to store changes. Compare motion quality, size, and decoding time. |
| 30 | Final Experiment: Choosing a Method for a Task | Prepare several versions of one scene. Compare quality, size, speed, and hardware requirements; justify the choice and save the settings to reproduce the experiment. |

**Result:** a scene with measured characteristics and a reproducible optimization experiment.

### Supplementary Materials

Outside the core route: the ecosystem and current papers, surface and mesh reconstruction, the 2DGS method, and advanced GPU programming. These topics are added when a particular task calls for them.

---

## How to Go Through This

Follow the order and run the examples. Each block builds on the previous one and ends with a working result.

There is no need to memorize a formula before its meaning is clear. If a symbol or operation is unclear, return to concrete numbers and the image, work through the missing step, and then continue.

Change one parameter at a time at first. Predict what will happen, test it, and explain any difference from your expectations. Then combine changes. Save intermediate results: you will use them to compare quality, speed, and size.

---

## Author

**Heinrich Wirth**

[@wirth_heinrich](https://x.com/wirth_heinrich)

---

## License

MIT.
