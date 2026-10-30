# Gaussian Splatting — interactive course

Nine lessons in Russian and English, with browser experiments, worked mathematics, and the original Python calculations. The canonical materials live in the repository's [`notebooks/`](../notebooks/); this directory contains their browser presentation.

## Local site

From the repository root:

```sh
python -m http.server 8000 --directory site/dist
```

The site is available at [http://localhost:8000/](http://localhost:8000/). A static HTTP server is needed because the browser loads ES modules and lesson JSON files. Opening `index.html` directly through `file://` does not provide that environment.

The complete site is in `dist/`. It needs no backend, Python kernel, API keys, or external CDN. The live experiments run JavaScript locally in the browser. Python disclosures show the notebook examples and their saved results; they do not execute Python in the page.

## Content generation

The exporter reads the eighteen notebooks from the same repository, adapts their presentation to the browser controls, and writes lesson JSON and saved figures into `site/dist/`. Python 3 and its standard library are sufficient:

```sh
python site/scripts/export_content.py
```

The equivalent npm command, also from the repository root, is:

```sh
npm --prefix site run build
```

The exporter uses existing notebook outputs. Changes to Python examples or illustration code require refreshed notebook outputs first, following the course's [notebook build instructions](../README.md). Committed site content is ready to serve without regenerating it.

The browser experiments are in `dist/labs/`; `dist/math.js` contains their Gaussian and alpha-composition calculations, while `dist/learning-math.js` contains loss functions, gradients, and the browser training loop. The second block uses actual projected gradient descent; PyTorch examples remain available in the notebooks. Paths are relative to the site directory, and lesson routes use hashes such as `#/ru/2`. The same files work at the GitHub Pages project path `/gaussian-splatting-course/`.

## Validation

Tests use Node.js 20 or newer and Python 3. Dependencies are bundled, so no `npm install` step is needed. From the repository root:

```sh
npm --prefix site test
```

The suite checks numerical examples, Gaussian symmetry, covariance/inverse equivalence, composition order and weights, local assets, every formula with KaTeX, and the exporter's lesson and README links. README links with section anchors point to the matching file on GitHub; ordinary course navigation stays inside the site.

## Libraries

KaTeX 0.16.22 and marked 17.0.5 are bundled with their MIT licenses in `dist/vendor/`.
