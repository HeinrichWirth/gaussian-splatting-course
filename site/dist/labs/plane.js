import * as gaussianMath from '../math.js';
import { makeLab } from './shared.js';

const COLORS = {
  ink: '#23323e', muted: '#63747f', border: '#d9e3e7', paper: '#f8fafb',
  teal: '#087f83', orange: '#c94d22', violet: '#7255b1', green: '#32865e',
};
let nextId = 0;
const text = (lang, ru, en) => lang === 'ru' ? ru : en;
const clamp = (value, lo, hi) => Math.max(lo, Math.min(hi, value));

function fmt(value, digits = 4) {
  if (!Number.isFinite(value)) return '—';
  if (value === 0) return '0';
  if (Math.abs(value) < 1e-4) return value.toExponential(3).replace('e-', 'e−');
  return Number(value.toFixed(digits)).toString().replace('-', '−');
}

function element(tag, className, content) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (content !== undefined) node.textContent = content;
  return node;
}

function installStyles() {
  const href = new URL('./plane.css', import.meta.url).href;
  if (!document.querySelector('link[data-plane-labs]')) {
    const link = document.createElement('link');
    link.rel = 'stylesheet';
    link.href = href;
    link.dataset.planeLabs = '';
    document.head.append(link);
  }
}

function heatColor(weight) {
  // Every view shares this absolute 0..1 scale; no per-frame normalization.
  const stops = [[19, 35, 62], [51, 65, 117], [51, 116, 146], [78, 179, 164], [241, 225, 137]];
  const position = clamp(weight, 0, 1) * (stops.length - 1);
  const index = Math.min(stops.length - 2, Math.floor(position));
  const t = position - index;
  return stops[index].map((v, k) => Math.round(v + t * (stops[index + 1][k] - v)));
}

function colorString(weight) { return `rgb(${heatColor(weight).join(',')})`; }

function canvasSize(canvas, heightRatio = 1) {
  const width = Math.max(240, Math.round(canvas.getBoundingClientRect().width || 440));
  const height = Math.round(width * heightRatio);
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  if (canvas.width !== Math.round(width * dpr) || canvas.height !== Math.round(height * dpr)) {
    canvas.width = Math.round(width * dpr);
    canvas.height = Math.round(height * dpr);
  }
  const ctx = canvas.getContext('2d');
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, width, height);
  ctx.fillStyle = COLORS.paper;
  ctx.fillRect(0, 0, width, height);
  return { ctx, width, height };
}

function line(ctx, points, color, width = 1, dash = []) {
  ctx.beginPath();
  points.forEach(([x, y], i) => i ? ctx.lineTo(x, y) : ctx.moveTo(x, y));
  ctx.strokeStyle = color;
  ctx.lineWidth = width;
  ctx.setLineDash(dash);
  ctx.stroke();
  ctx.setLineDash([]);
}

function label(ctx, value, x, y, color = COLORS.muted, align = 'center', size = 12) {
  ctx.font = `500 ${size}px system-ui, sans-serif`;
  ctx.textAlign = align;
  ctx.textBaseline = 'middle';
  ctx.fillStyle = color;
  ctx.fillText(value, x, y);
}

function marker(ctx, point, color = COLORS.orange, radius = 6) {
  ctx.beginPath();
  ctx.arc(point[0], point[1], radius, 0, Math.PI * 2);
  ctx.fillStyle = color;
  ctx.fill();
  ctx.strokeStyle = '#fff';
  ctx.lineWidth = 2;
  ctx.stroke();
}

function cross(ctx, point, color = '#fff', size = 6) {
  line(ctx, [[point[0] - size, point[1]], [point[0] + size, point[1]]], '#23323e', 4);
  line(ctx, [[point[0], point[1] - size], [point[0], point[1] + size]], '#23323e', 4);
  line(ctx, [[point[0] - size, point[1]], [point[0] + size, point[1]]], color, 2);
  line(ctx, [[point[0], point[1] - size], [point[0], point[1] + size]], color, 2);
}

function arrow(ctx, from, to, color, width = 2) {
  line(ctx, [from, to], color, width);
  const a = Math.atan2(to[1] - from[1], to[0] - from[0]);
  line(ctx, [[to[0] - 8 * Math.cos(a - .45), to[1] - 8 * Math.sin(a - .45)], to,
    [to[0] - 8 * Math.cos(a + .45), to[1] - 8 * Math.sin(a + .45)]], color, width);
}

function planeTransform(width, height, limit) {
  const padding = { left: 38, top: 18, right: 18, bottom: 76 };
  const side = Math.min(width - padding.left - padding.right, height - padding.top - padding.bottom);
  const left = padding.left + (width - padding.left - padding.right - side) / 2;
  const top = padding.top;
  return { left, top, side,
    toPixel: (x, y) => [left + (x + limit) * side / (2 * limit), top + (limit - y) * side / (2 * limit)],
    fromPixel: (px, py) => [(px - left) * 2 * limit / side - limit, limit - (py - top) * 2 * limit / side],
  };
}

function drawHeatmap(canvas, model, evaluate, lang, options = {}) {
  const { ctx, width, height } = canvasSize(canvas, 1.08);
  const limit = options.limit || 5;
  const plot = planeTransform(width, height, limit);
  const resolution = 181;
  const buffer = document.createElement('canvas');
  buffer.width = buffer.height = resolution;
  const imageCtx = buffer.getContext('2d');
  const image = imageCtx.createImageData(resolution, resolution);
  for (let row = 0; row < resolution; row++) {
    const y = limit - row * 2 * limit / (resolution - 1);
    for (let col = 0; col < resolution; col++) {
      const x = -limit + col * 2 * limit / (resolution - 1);
      const rgb = heatColor(evaluate(x, y));
      const index = 4 * (row * resolution + col);
      image.data.set([...rgb, 255], index);
    }
  }
  imageCtx.putImageData(image, 0, 0);
  ctx.drawImage(buffer, plot.left, plot.top, plot.side, plot.side);
  ctx.save();
  ctx.beginPath();
  ctx.rect(plot.left, plot.top, plot.side, plot.side);
  ctx.clip();
  for (let tick = -limit; tick <= limit; tick++) {
    line(ctx, [plot.toPixel(tick, -limit), plot.toPixel(tick, limit)], tick === 0 ? '#ffffff6b' : '#ffffff1a');
    line(ctx, [plot.toPixel(-limit, tick), plot.toPixel(limit, tick)], tick === 0 ? '#ffffff6b' : '#ffffff1a');
  }
  const angle = model.angle * Math.PI / 180;
  const c = Math.cos(angle), s = Math.sin(angle);
  for (const radius of [2, 1]) {
    const points = Array.from({ length: 181 }, (_, i) => {
      const t = i * 2 * Math.PI / 180;
      const a = radius * model.sx * Math.cos(t), b = radius * model.sy * Math.sin(t);
      return plot.toPixel(model.mux + c * a - s * b, model.muy + s * a + c * b);
    });
    line(ctx, points, radius === 1 ? '#fff' : '#ffffffa6', radius === 1 ? 2 : 1.25, radius === 1 ? [] : [4, 4]);
  }
  if (options.axes) {
    const center = plot.toPixel(model.mux, model.muy);
    const a = plot.toPixel(model.mux + model.sx * c, model.muy + model.sx * s);
    const b = plot.toPixel(model.mux - model.sy * s, model.muy + model.sy * c);
    arrow(ctx, center, a, '#ffb06a', 3);
    arrow(ctx, center, b, '#8ff1bb', 3);
    label(ctx, 's₁', a[0] + 9, a[1] - 10, '#ffcf9b', 'left', 14);
    label(ctx, 's₂', b[0] + 9, b[1] - 10, '#b8ffd6', 'left', 14);
  }
  if (options.slices) {
    line(ctx, [plot.toPixel(-limit, model.y), plot.toPixel(limit, model.y)], '#94e3ef', 2, [6, 4]);
    line(ctx, [plot.toPixel(model.x, -limit), plot.toPixel(model.x, limit)], '#ffb06a', 2, [6, 4]);
  }
  cross(ctx, plot.toPixel(model.mux, model.muy));
  marker(ctx, plot.toPixel(model.x, model.y));
  ctx.restore();
  ctx.strokeStyle = COLORS.border;
  ctx.strokeRect(plot.left, plot.top, plot.side, plot.side);
  for (const tick of [-limit, -limit / 2, 0, limit / 2, limit]) {
    const [px] = plot.toPixel(tick, 0);
    const [, py] = plot.toPixel(0, tick);
    label(ctx, fmt(tick), px, plot.top + plot.side + 15);
    label(ctx, fmt(tick), plot.left - 8, py, COLORS.muted, 'right');
  }
  label(ctx, 'x', plot.left + plot.side + 9, plot.top + plot.side + 15);
  label(ctx, 'y', plot.left - 16, plot.top - 7);
  const barY = plot.top + plot.side + 42;
  const gradient = ctx.createLinearGradient(plot.left, 0, plot.left + plot.side, 0);
  for (let k = 0; k <= 20; k++) gradient.addColorStop(k / 20, colorString(k / 20));
  ctx.fillStyle = gradient;
  ctx.fillRect(plot.left, barY, plot.side, 9);
  label(ctx, '0', plot.left, barY + 20, COLORS.muted, 'left');
  label(ctx, text(lang, 'Вес · 0 … 1', 'Weight · 0 … 1'), plot.left + plot.side / 2, barY + 20);
  label(ctx, '1', plot.left + plot.side, barY + 20, COLORS.muted, 'right');
  return plot;
}

function drawHill(canvas, model, evaluate, lang) {
  const { ctx, width, height } = canvasSize(canvas, 1.08);
  const limit = 5;
  const unit = (width - 60) / (4 * limit);
  const zScale = height * .38;
  const project = (x, y, z = 0) => [width / 2 + (x - y) * unit, height * .78 - (x + y + 2 * limit) * unit * .48 - z * zScale];
  for (let t = -limit; t <= limit; t++) {
    line(ctx, [project(t, -limit), project(t, limit)], COLORS.border);
    line(ctx, [project(-limit, t), project(limit, t)], COLORS.border);
  }
  const n = 40, cells = [];
  const baseGrid = Array.from({ length: n + 1 }, (_, i) => -limit + i * 2 * limit / n);
  // Explicit center/probe samples keep the peak at one during translation,
  // and keep the orange marker on the actual surface at narrow scales.
  const xs = [...new Set([...baseGrid, model.mux, model.x])].sort((a, b) => a - b);
  const ys = [...new Set([...baseGrid, model.muy, model.y])].sort((a, b) => a - b);
  for (let i = 0; i < xs.length - 1; i++) {
    for (let j = 0; j < ys.length - 1; j++) {
      const x = xs[i], y = ys[j];
      const xy = [[x, y], [xs[i + 1], y], [xs[i + 1], ys[j + 1]], [x, ys[j + 1]]];
      const weights = xy.map(([a, b]) => evaluate(a, b));
      cells.push({ depth: x + y, xy, weights });
    }
  }
  cells.sort((a, b) => b.depth - a.depth);
  for (const cell of cells) {
    ctx.beginPath();
    cell.xy.forEach(([x, y], i) => {
      const point = project(x, y, cell.weights[i]);
      i ? ctx.lineTo(...point) : ctx.moveTo(...point);
    });
    ctx.closePath();
    ctx.fillStyle = colorString(cell.weights.reduce((a, b) => a + b) / 4);
    ctx.fill();
    ctx.strokeStyle = '#ffffff12';
    ctx.lineWidth = .5;
    ctx.stroke();
  }
  const origin = project(-limit, -limit);
  const zBase = project(-limit, limit, 0);
  const zTop = project(-limit, limit, 1);
  arrow(ctx, origin, project(limit, -limit), COLORS.muted);
  arrow(ctx, origin, project(-limit, limit), COLORS.muted);
  arrow(ctx, zBase, zTop, COLORS.muted);
  label(ctx, 'x', ...project(limit + .6, -limit));
  const yEnd = project(-limit, limit);
  label(ctx, 'y', yEnd[0] - 14, yEnd[1] + 33);
  label(ctx, text(lang, 'Вес', 'Weight'), zTop[0], zTop[1] - 15, COLORS.muted, 'left');
  label(ctx, '0', zBase[0] - 10, zBase[1], COLORS.muted, 'right');
  label(ctx, '1', zTop[0] - 10, zTop[1], COLORS.muted, 'right');
  label(ctx, '(−5, −5)', origin[0], origin[1] + 17, COLORS.muted, 'center', 11);
  for (const tick of [0, 5]) {
    let p = project(tick, -limit);
    label(ctx, String(tick).replace('-', '−'), p[0] + 7, p[1] + 15);
    p = project(-limit, tick);
    label(ctx, String(tick).replace('-', '−'), p[0] - 7, p[1] + 15);
  }
  const weight = evaluate(model.x, model.y);
  const base = project(model.x, model.y), top = project(model.x, model.y, weight);
  line(ctx, [base, top], COLORS.orange, 2, [4, 3]);
  marker(ctx, top);
  label(ctx, `f = ${fmt(weight, 6)}`, width / 2, height - 31, COLORS.ink, 'center', 14);
  label(ctx, text(lang, 'Высота — вес; это не координата z предмета.', 'Height is weight, not the object’s z coordinate.'), width / 2, height - 11, COLORS.muted, 'center', width < 350 ? 10 : 11);
}

function drawSlices(canvas, model, evaluate, lang) {
  const { ctx, width, height } = canvasSize(canvas, 1.08);
  const left = 42, right = width - 20;
  const panels = [
    { axis: 'x', fixed: model.y, center: model.mux, sample: model.x, y0: 30, color: COLORS.teal,
      actual: x => evaluate(x, model.y), reference: x => evaluate(x, model.muy) },
    { axis: 'y', fixed: model.x, center: model.muy, sample: model.y, y0: height / 2 + 20, color: COLORS.orange,
      actual: y => evaluate(model.x, y), reference: y => evaluate(model.mux, y) },
  ];
  const ph = height / 2 - 86;
  for (const panel of panels) {
    const px = x => left + (x + 5) / 10 * (right - left);
    const py = w => panel.y0 + ph * (1 - w);
    const fixedAxis = panel.axis === 'x' ? 'y' : 'x';
    label(ctx, `${fixedAxis} = ${fmt(panel.fixed)}`, left, panel.y0 - 15, COLORS.ink, 'left', 13);
    for (const w of [0, .5, 1]) {
      line(ctx, [[left, py(w)], [right, py(w)]], COLORS.border);
      label(ctx, fmt(w), left - 7, py(w), COLORS.muted, 'right', 11);
    }
    for (const tick of [-5, 0, 5]) label(ctx, String(tick).replace('-', '−'), px(tick), py(0) + 15);
    label(ctx, panel.axis, right + 10, py(0) + 15);
    const points = [...new Set([...Array.from({ length: 241 }, (_, i) => -5 + i / 24), panel.center, panel.sample])].sort((a, b) => a - b);
    line(ctx, points.map(v => [px(v), py(panel.reference(v))]), '#adb9c0', 1.8, [5, 4]);
    line(ctx, points.map(v => [px(v), py(panel.actual(v))]), panel.color, 2.7);
    const w = panel.actual(panel.sample);
    line(ctx, [[px(panel.sample), py(0)], [px(panel.sample), py(w)]], COLORS.orange, 1, [3, 3]);
    marker(ctx, [px(panel.sample), py(w)], COLORS.orange, 5);
    label(ctx, `${text(lang, 'Вершина среза', 'Slice peak')} = ${fmt(panel.actual(panel.center), 6)}`,
      left, py(0) + 37, COLORS.muted, 'left', width < 350 ? 10 : 11);
  }
  line(ctx, [[left, height - 10], [left + 23, height - 10]], '#adb9c0', 1.8, [5, 4]);
  label(ctx, text(lang, 'Пунктир — срез через центр', 'Dashed: slice through the center'), left + 31, height - 10, COLORS.muted, 'left', 11);
}

function stepCard(parent, title, equations, note = '') {
  const card = element('section', 'plab-step');
  card.append(element('h4', '', title));
  for (const equation of equations) card.append(element('div', 'plab-equation', equation));
  if (note) card.append(element('p', 'plab-step-note', note));
  parent.append(card);
  return card;
}

function matrixCard(parent, name, matrix, description) {
  const card = element('section', 'plab-matrix-card');
  const title = element('h4', '', name);
  const table = element('table', 'plab-matrix');
  table.setAttribute('aria-label', description);
  const body = document.createElement('tbody');
  matrix.forEach(row => {
    const tr = document.createElement('tr');
    row.forEach(value => tr.append(element('td', '', fmt(value, 5))));
    body.append(tr);
  });
  table.append(body);
  card.append(title, table, element('p', 'plab-step-note', description));
  parent.append(card);
}

function planeReadout(parent, model, steps, lang) {
  parent.replaceChildren();
  const grid = element('div', 'plab-steps');
  const wx = Math.exp(-(steps.u ** 2) / 2), wy = Math.exp(-(steps.v ** 2) / 2);
  stepCard(grid, text(lang, '1 · От центра', '1 · From the center'), [
    `dₓ = ${fmt(model.x)} − (${fmt(model.mux)}) = ${fmt(steps.dx)}`,
    `dᵧ = ${fmt(model.y)} − (${fmt(model.muy)}) = ${fmt(steps.dy)}`,
  ], text(lang, 'Смещения в единицах координат.', 'Offsets in coordinate units.'));
  stepCard(grid, text(lang, '2 · В единицах ширины', '2 · In scale units'), [
    `uₓ = dₓ / σₓ = ${fmt(steps.dx)} / ${fmt(model.sx)} ≈ ${fmt(steps.u, 6)}`,
    `uᵧ = dᵧ / σᵧ = ${fmt(steps.dy)} / ${fmt(model.sy)} ≈ ${fmt(steps.v, 6)}`,
  ], text(lang, 'Единицы длины сокращаются.', 'The length units cancel.'));
  stepCard(grid, text(lang, '3 · Два квадрата', '3 · Two squares'), [
    `uₓ² ≈ (${fmt(steps.u, 6)})² ≈ ${fmt(steps.u ** 2, 6)}`,
    `uᵧ² ≈ (${fmt(steps.v, 6)})² ≈ ${fmt(steps.v ** 2, 6)}`,
  ], text(lang, 'Знак смещения исчезает; удаление остаётся.', 'The signs disappear; the separation remains.'));
  stepCard(grid, text(lang, '4 · По одному весу на ось', '4 · One weight per axis'), [
    `wₓ = exp(−uₓ² / 2) ≈ ${fmt(wx, 6)}`,
    `wᵧ = exp(−uᵧ² / 2) ≈ ${fmt(wy, 6)}`,
  ], text(lang, 'Каждая ось даёт одномерный колокол.', 'Each axis contributes a one-dimensional bell.'));
  stepCard(grid, text(lang, '5 · Сумма внутри экспоненты', '5 · A sum inside the exponential'), [
    `q = uₓ² + uᵧ² ≈ ${fmt(steps.q, 6)}`,
    `−q / 2 ≈ ${fmt(-steps.q / 2, 6)}`,
  ], text(lang, 'При умножении eᵃ · eᵇ = eᵃ⁺ᵇ показатели складываются.', 'In eᵃ · eᵇ = eᵃ⁺ᵇ, multiplication adds the exponents.'));
  stepCard(grid, text(lang, '6 · Один результат двумя путями', '6 · Two routes, one result'), [
    `f = wₓ · wᵧ ≈ ${fmt(wx * wy, 6)}`,
    `f = exp(−q / 2) ≈ ${fmt(steps.weight, 6)}`,
  ], text(lang, 'Это вес на карте, высота холма и значение обоих срезов в точке.', 'This is the map’s weight, the hill’s height, and both slices’ value at the point.'));
  parent.append(grid);
}

function covarianceReadout(parent, model, steps, lang) {
  parent.replaceChildren();
  const angle = model.angle * Math.PI / 180, c = Math.cos(angle), s = Math.sin(angle);
  const matrix = steps.covariance, inverse = steps.inverse;
  const a = matrix[0][0], b = matrix[0][1], d = matrix[1][1];
  const determinant = a * d - b * b;
  const matrices = element('div', 'plab-matrices');
  matrixCard(matrices, 'R', [[c, -s], [s, c]], text(lang, 'Поворот собственных осей в координаты x, y.', 'Local axes rotated into x, y coordinates.'));
  matrixCard(matrices, 'Σ = R · diag(s₁², s₂²) · Rᵀ', matrix, text(lang, 'Форма пятна; элементы в единицах длины².', 'The spot’s shape; entries have units of length².'));
  matrixCard(matrices, 'Σ⁻¹', inverse, text(lang, 'Обратная матрица; элементы в 1 / длина².', 'Inverse matrix; entries have units of 1 / length².'));
  const grid = element('div', 'plab-steps');
  stepCard(grid, text(lang, '1 · Поворот точки назад', '1 · Rotating the point back'), [
    `c = cos(${fmt(model.angle)}°) ≈ ${fmt(c, 6)}`,
    `s = sin(${fmt(model.angle)}°) ≈ ${fmt(s, 6)}`,
    `d₁ = c·x + s·y ≈ (${fmt(c)})·(${fmt(model.x)}) + (${fmt(s)})·(${fmt(model.y)}) ≈ ${fmt(steps.localX, 6)}`,
    `d₂ = −s·x + c·y ≈ (${fmt(-s)})·(${fmt(model.x)}) + (${fmt(c)})·(${fmt(model.y)}) ≈ ${fmt(steps.localY, 6)}`,
  ], text(lang, 'Центр (0, 0). Координаты d₁, d₂ идут вдоль осей пятна.', 'The center is (0, 0). Coordinates d₁, d₂ follow the spot’s axes.'));
  stepCard(grid, text(lang, '2 · Масштаб каждой оси', '2 · Scaling each axis'), [
    `u₁ = d₁ / s₁ ≈ ${fmt(steps.localX, 6)} / ${fmt(model.sx)} ≈ ${fmt(steps.u, 6)}`,
    `u₂ = d₂ / s₂ ≈ ${fmt(steps.localY, 6)} / ${fmt(model.sy)} ≈ ${fmt(steps.v, 6)}`,
    `q = u₁² + u₂² ≈ (${fmt(steps.u)})² + (${fmt(steps.v)})² ≈ ${fmt(steps.q, 6)}`,
    `f = exp(−q / 2) ≈ exp(${fmt(-steps.q / 2, 6)}) ≈ ${fmt(steps.weight, 6)}`,
  ], text(lang, 'Тот же расчёт, что у неповёрнутого пятна, в его собственных координатах.', 'The same calculation as before, in the spot’s own coordinates.'));
  stepCard(grid, text(lang, '3 · От масштаба к матрице', '3 · From scales to a matrix'), [
    `a = c²s₁² + s²s₂² ≈ ${fmt(a, 6)}`,
    `b = cs(s₁² − s₂²) ≈ ${fmt(b, 6)}`,
    `d = s²s₁² + c²s₂² ≈ ${fmt(d, 6)}`,
    `Σ = [[a, b], [b, d]]`,
  ], text(lang, 'Когда s₁ = s₂, b = 0 при любом угле: круг не меняет форму.', 'When s₁ = s₂, b = 0 at every angle: the circle’s shape stays the same.'));
  stepCard(grid, text(lang, '4 · Обратная матрица', '4 · The inverse matrix'), [
    `det Σ = a·d − b² ≈ (${fmt(a)})·(${fmt(d)}) − (${fmt(b)})² ≈ ${fmt(determinant, 6)}`,
    `Σ⁻¹ = [[d, −b], [−b, a]] / det Σ`,
    `Σ⁻¹₁₁ ≈ ${fmt(d)} / ${fmt(determinant)} ≈ ${fmt(inverse[0][0], 6)}`,
    `Σ⁻¹₁₂ = Σ⁻¹₂₁ ≈ ${fmt(-b)} / ${fmt(determinant)} ≈ ${fmt(inverse[0][1], 6)}`,
    `Σ⁻¹₂₂ ≈ ${fmt(a)} / ${fmt(determinant)} ≈ ${fmt(inverse[1][1], 6)}`,
  ], text(lang, 'При положительных масштабах det Σ = s₁²s₂² > 0, деление допустимо.', 'Positive scales give det Σ = s₁²s₂² > 0, so the division is defined.'));
  const vx = inverse[0][0] * model.x + inverse[0][1] * model.y;
  const vy = inverse[1][0] * model.x + inverse[1][1] * model.y;
  const qMatrix = model.x * vx + model.y * vy;
  stepCard(grid, text(lang, '5 · Тот же q через матрицу', '5 · The same q through the matrix'), [
    `v₁ = Σ⁻¹₁₁·x + Σ⁻¹₁₂·y ≈ ${fmt(vx, 6)}`,
    `v₂ = Σ⁻¹₂₁·x + Σ⁻¹₂₂·y ≈ ${fmt(vy, 6)}`,
    `q = x·v₁ + y·v₂ ≈ (${fmt(model.x)})·(${fmt(vx)}) + (${fmt(model.y)})·(${fmt(vy)}) ≈ ${fmt(qMatrix, 6)}`,
    `f = exp(−q / 2) ≈ ${fmt(Math.exp(-qMatrix / 2), 6)}`,
  ], text(lang, 'Матрица и поворот точки дают одинаковый вес. v₁ и v₂ — временные результаты умножения.', 'The matrix and the rotated point give the same weight. v₁ and v₂ are intermediate products.'));
  parent.append(matrices, grid);
}

const PLANE_DEFAULT = { mux: 0, muy: 0, sx: 2, sy: 1, x: 2, y: 1, view: 'hill', preset: 'default' };
const COV_DEFAULT = { s1: 2, s2: 1, angle: 45, x: 1, y: 1, preset: 'default' };
const PLANE_PRESETS = {
  default: { mux: 0, muy: 0, sx: 2, sy: 1, x: 2, y: 1 },
  circle: { mux: 0, muy: 0, sx: 1, sy: 1, x: 1, y: 0 },
  vertical: { mux: 0, muy: 0, sx: 1, sy: 2, x: 1, y: 2 },
  one_width: { mux: 0, muy: 0, sx: 2, sy: 1, x: 2, y: 0 },
  center: { mux: 0, muy: 0, sx: 2, sy: 1, x: 0, y: 0 },
};
const COV_PRESETS = {
  default: { s1: 2, s2: 1, angle: 45, x: 1, y: 1 },
  axis: { s1: 2, s2: 1, angle: 0, x: 1, y: 1 },
  opposite: { s1: 2, s2: 1, angle: -45, x: 1, y: 1 },
  circle: { s1: 1.5, s2: 1.5, angle: 45, x: 1, y: 1 },
  center: { s1: 2, s2: 1, angle: 45, x: 0, y: 0 },
};

function mount(container, language, initial, kind) {
  const lang = language === 'en' ? 'en' : 'ru';
  const covariance = kind === 'covariance';
  const defaults = covariance ? COV_DEFAULT : PLANE_DEFAULT;
  const presets = covariance ? COV_PRESETS : PLANE_PRESETS;
  const state = { ...defaults };
  const specifications = covariance ? [
    ['s1', .25, 3, .05, 'Масштаб s₁', 'Scale s₁'],
    ['s2', .25, 3, .05, 'Масштаб s₂', 'Scale s₂'],
    ['angle', -180, 180, 1, 'Поворот θ, °', 'Rotation θ, °'],
    ['x', -4, 4, .05, 'Точка: x', 'Point: x'],
    ['y', -4, 4, .05, 'Точка: y', 'Point: y'],
  ] : [
    ['mux', -2, 2, .05, 'Центр μₓ', 'Center μₓ'],
    ['muy', -2, 2, .05, 'Центр μᵧ', 'Center μᵧ'],
    ['sx', .3, 2.5, .05, 'Ширина σₓ', 'Scale σₓ'],
    ['sy', .3, 2.5, .05, 'Ширина σᵧ', 'Scale σᵧ'],
    ['x', -4, 4, .05, 'Точка: x', 'Point: x'],
    ['y', -4, 4, .05, 'Точка: y', 'Point: y'],
  ];
  const source = initial && typeof initial === 'object' ? initial : {};
  if (source.preset && presets[source.preset]) Object.assign(state, presets[source.preset], { preset: source.preset });
  for (const [key, lo, hi] of specifications) {
    const value = source[key] ?? (key === 'angle' ? source.angle_deg : undefined);
    if (typeof value === 'number' && Number.isFinite(value)) state[key] = clamp(value, lo, hi);
  }
  if (!covariance && ['hill', 'slices'].includes(source.view)) state.view = source.view;
  if (!Object.entries(presets[state.preset]).every(([key, value]) => state[key] === value)) state.preset = 'custom';
  installStyles();
  const lab = makeLab(container, {
    title: covariance ? text(lang, 'Поворот, матрица и одна точка', 'A rotation, a matrix, and one point') : text(lang, 'Одна функция на плоскости', 'One function on a plane'),
    subtitle: covariance ? text(lang, 'Центр закреплён в (0, 0). Поворот меняет собственные оси и числа матрицы.', 'The center stays at (0, 0). Rotation changes the local axes and the matrix entries.') : text(lang, 'Цвет, высота и срезы показывают один и тот же вес.', 'Color, height, and slices show the same weight.'),
    lang, tag: text(lang, 'ОПЫТ', 'EXPERIMENT'),
  });
  lab.root.classList.add('plab', covariance ? 'plab-covariance' : 'plab-plane');
  lab.root.dataset.lab = kind;
  const lifecycle = new AbortController();
  const on = (node, event, handler) => node.addEventListener(event, handler, { signal: lifecycle.signal });
  let frame = 0, destroyed = false, transform = null;
  const controls = {};
  const plots = element('div', covariance ? 'plab-plots plab-plots-single' : 'plab-plots');
  const figure = (title, description) => {
    const section = element('figure', 'plab-plot');
    section.append(element('figcaption', 'plab-plot-title', title));
    const canvas = element('canvas', 'plab-canvas');
    canvas.setAttribute('role', 'img');
    canvas.setAttribute('aria-label', description);
    section.append(canvas);
    plots.append(section);
    return { section, canvas };
  };
  const heat = figure(text(lang, 'Вес на карте', 'Weight on the map'), text(lang, 'Тепловая карта гауссианы с центром, выбранной точкой и линиями одинакового веса.', 'Gaussian heatmap with its center, selected point, and equal-weight contours.'));
  const companion = covariance ? null : figure(text(lang, 'Вес как высота', 'Weight as height'), text(lang, 'График той же функции: высота от 0 до 1, координаты от −5 до 5.', 'The same function: height from 0 to 1, coordinates from −5 to 5.'));
  lab.view.append(plots);
  const result = element('output', 'plab-result');
  result.setAttribute('aria-live', 'polite');
  result.setAttribute('aria-atomic', 'true');
  result.dataset.testid = 'point-weight';
  lab.view.append(result);
  lab.view.append(element('p', 'plab-legend', text(lang,
    'Белый плюс — центр. Оранжевая точка — место расчёта. Сплошной контур: q = 1, f ≈ 0.607; пунктир: q = 4, f ≈ 0.135.',
    'White plus: center. Orange point: calculation location. Solid contour: q = 1, f ≈ 0.607; dashed: q = 4, f ≈ 0.135.')));
  const controlHeading = element('h4', 'plab-control-heading', text(lang, 'Параметры', 'Parameters'));
  lab.controls.append(controlHeading);

  function schedule() {
    if (destroyed || frame) return;
    frame = requestAnimationFrame(() => { frame = 0; render(); });
  }
  function markCustom() {
    state.preset = 'custom';
    if (controls.preset) controls.preset.value = 'custom';
  }
  function change(key, value) {
    const spec = specifications.find(item => item[0] === key);
    if (!Number.isFinite(value) || !spec) return;
    state[key] = clamp(value, spec[1], spec[2]);
    markCustom();
    sync();
    schedule();
  }
  for (const [key, lo, hi, step, ru, en] of specifications) {
    const labelText = text(lang, ru, en);
    const row = element('div', 'plab-control');
    const labelId = `plab-label-${++nextId}`;
    const top = element('div', 'plab-control-top');
    const name = element('label', '', labelText);
    name.id = labelId;
    const range = element('input', 'plab-range');
    range.type = 'range';
    range.id = `plab-range-${nextId}`;
    name.htmlFor = range.id;
    Object.assign(range, { min: lo, max: hi, step, value: state[key] });
    range.dataset.key = key;
    range.setAttribute('aria-labelledby', labelId);
    const number = element('input', 'plab-number');
    number.type = 'number';
    Object.assign(number, { min: lo, max: hi, step, value: state[key] });
    number.setAttribute('aria-label', `${labelText} — ${text(lang, 'число', 'number')}`);
    number.dataset.key = `${key}-number`;
    top.append(name, number);
    row.append(top, range);
    lab.controls.append(row);
    controls[key] = { range, number };
    on(range, 'input', () => change(key, Number(range.value)));
    on(number, 'input', () => {
      const value = number.valueAsNumber;
      // Empty, minus-only and out-of-range text can be intermediate edits.
      // Valid input updates the plot without rewriting the active text field.
      if (!number.value.trim() || !Number.isFinite(value) || value < lo || value > hi) return;
      state[key] = value;
      markCustom();
      sync(key);
      schedule();
    });
    const normalizeNumber = () => {
      if (number.value.trim() && Number.isFinite(number.valueAsNumber)) change(key, number.valueAsNumber);
      else sync();
    };
    on(number, 'change', normalizeNumber);
    on(number, 'blur', normalizeNumber);
  }
  const presetRow = element('label', 'plab-select-row');
  presetRow.append(element('span', '', text(lang, 'Пример', 'Example')));
  const preset = element('select', 'lab-select');
  preset.dataset.key = 'preset';
  preset.setAttribute('aria-label', text(lang, 'Пример', 'Example'));
  const options = covariance ? [
    ['default', 'Поворот 45°', 'A 45° rotation'], ['axis', 'Оси совпадают', 'Aligned axes'],
    ['opposite', 'Поворот −45°', 'A −45° rotation'], ['circle', 'Круг', 'Circle'], ['center', 'Точка в центре', 'Point at the center'],
  ] : [
    ['default', 'Исходное пятно', 'Starting spot'], ['circle', 'Круг', 'Circle'], ['vertical', 'Вытянуто по y', 'Stretched along y'],
    ['one_width', 'На одной ширине', 'One scale away'], ['center', 'Точка в центре', 'Point at the center'],
  ];
  options.push(['custom', 'Изменённые параметры', 'Modified parameters']);
  for (const [value, ru, en] of options) {
    const option = element('option', '', text(lang, ru, en));
    option.value = value;
    preset.append(option);
  }
  presetRow.append(preset);
  lab.controls.append(presetRow);
  controls.preset = preset;
  on(preset, 'change', () => {
    if (presets[preset.value]) Object.assign(state, presets[preset.value], { preset: preset.value });
    sync();
    schedule();
  });
  if (!covariance) {
    const viewRow = element('label', 'plab-select-row');
    viewRow.append(element('span', '', text(lang, 'Второй вид', 'Second view')));
    const view = element('select', 'lab-select');
    view.setAttribute('aria-label', text(lang, 'Второй вид', 'Second view'));
    view.dataset.key = 'view';
    for (const [value, ru, en] of [['hill', 'Холм', 'Hill'], ['slices', 'Два среза', 'Two slices']]) {
      const option = element('option', '', text(lang, ru, en));
      option.value = value;
      view.append(option);
    }
    viewRow.append(view);
    lab.controls.append(viewRow);
    controls.view = view;
    on(view, 'change', () => { state.view = view.value; sync(); schedule(); });
  }
  const reset = element('button', 'lab-button plab-reset', text(lang, 'Сбросить', 'Reset'));
  reset.type = 'button';
  reset.dataset.action = 'reset';
  lab.controls.append(reset);
  on(reset, 'click', () => { Object.assign(state, defaults); sync(); schedule(); });

  const canvasNote = element('p', 'plab-interaction-note', text(lang,
    'Положение точки задают ползунки, числовые поля или нажатие на карту. На карте доступны клавиши ← ↑ → ↓; Shift увеличивает шаг.',
    'Sliders, number fields, and a click on the map set the point. The map also supports ← ↑ → ↓; Shift increases the step.'));
  canvasNote.id = `plab-map-help-${++nextId}`;
  heat.section.append(canvasNote);
  heat.canvas.tabIndex = 0;
  heat.canvas.setAttribute('aria-describedby', canvasNote.id);
  heat.canvas.setAttribute('aria-keyshortcuts', 'ArrowLeft ArrowRight ArrowUp ArrowDown');
  on(heat.canvas, 'pointerdown', event => {
    if (!transform) return;
    const rect = heat.canvas.getBoundingClientRect();
    const px = event.clientX - rect.left, py = event.clientY - rect.top;
    if (px < transform.left || px > transform.left + transform.side || py < transform.top || py > transform.top + transform.side) return;
    const [x, y] = transform.fromPixel(px, py);
    state.x = Math.round(clamp(x, -4, 4) * 20) / 20;
    state.y = Math.round(clamp(y, -4, 4) * 20) / 20;
    markCustom(); sync(); schedule();
  });
  on(heat.canvas, 'keydown', event => {
    const direction = { ArrowLeft: [-1, 0], ArrowRight: [1, 0], ArrowUp: [0, 1], ArrowDown: [0, -1] }[event.key];
    if (!direction) return;
    event.preventDefault();
    const step = event.shiftKey ? .5 : .05;
    state.x = Math.round(clamp(state.x + direction[0] * step, -4, 4) * 1e8) / 1e8;
    state.y = Math.round(clamp(state.y + direction[1] * step, -4, 4) * 1e8) / 1e8;
    markCustom(); sync(); schedule();
  });
  lab.footer.textContent = covariance ? text(lang,
    'Координаты от −4 до 4, цвет от 0 до 1. Положительный угол — против часовой стрелки. Масштабы строго положительны; центр всегда (0, 0).',
    'Coordinates stay between −4 and 4; colors between 0 and 1. Positive angles turn counterclockwise. Scales are strictly positive; the center stays at (0, 0).') : text(lang,
    'Координаты от −5 до 5, цвет и высота от 0 до 1. Ширины строго положительны. Контуры показывают уровни веса, а не край пятна и не долю вероятности.',
    'Coordinates stay between −5 and 5; color and height between 0 and 1. Scales are strictly positive. Contours show weight levels, not the spot’s edge or a probability fraction.');

  function sync(keepNumber = null) {
    for (const [key] of specifications) {
      controls[key].range.value = state[key];
      if (key !== keepNumber) controls[key].number.value = Number(state[key].toFixed(6));
      controls[key].range.setAttribute('aria-valuetext', fmt(state[key], 6));
    }
    controls.preset.value = state.preset;
    if (controls.view) controls.view.value = state.view;
  }
  function render() {
    if (destroyed) return;
    const model = covariance ? { mux: 0, muy: 0, sx: state.s1, sy: state.s2, angle: state.angle, x: state.x, y: state.y }
      : { ...state, angle: 0 };
    const evaluate = (x, y) => gaussianMath.gaussian2D(x, y, [model.mux, model.muy], [model.sx, model.sy], model.angle);
    const steps = gaussianMath.gaussianSteps(model.x, model.y, [model.mux, model.muy], [model.sx, model.sy], model.angle);
    transform = drawHeatmap(heat.canvas, model, evaluate, lang, { limit: covariance ? 4 : 5, axes: covariance, slices: state.view === 'slices' });
    if (companion) {
      companion.section.querySelector('figcaption').textContent = state.view === 'hill' ? text(lang, 'Вес как высота', 'Weight as height') : text(lang, 'Два среза через точку', 'Two slices through the point');
      companion.canvas.setAttribute('aria-label', state.view === 'hill' ? text(lang, 'График гауссианы в виде холма; высота равна весу от 0 до 1.', 'A Gaussian hill with height equal to its weight, from 0 to 1.') : text(lang, 'Горизонтальный и вертикальный срезы; пунктирные линии проходят через центр.', 'Horizontal and vertical slices; dashed reference curves pass through the center.'));
      (state.view === 'hill' ? drawHill : drawSlices)(companion.canvas, model, evaluate, lang);
    }
    result.textContent = `f(${fmt(model.x)}, ${fmt(model.y)}) ≈ ${fmt(steps.weight, 6)}   ·   q ≈ ${fmt(steps.q, 6)}`;
    result.dataset.weight = String(steps.weight);
    result.dataset.q = String(steps.q);
    (covariance ? covarianceReadout : planeReadout)(lab.readout, model, steps, lang);
    heat.canvas.setAttribute('aria-label', `${text(lang, 'Гауссиана. Центр', 'Gaussian. Center')} (${fmt(model.mux)}, ${fmt(model.muy)}). ${text(lang, 'Точка', 'Point')} (${fmt(model.x)}, ${fmt(model.y)}). f ≈ ${fmt(steps.weight, 6)}.`);
    lab.root.dispatchEvent(new CustomEvent('labchange', { bubbles: true, detail: { ...state } }));
  }
  sync();
  render();
  const resize = typeof ResizeObserver === 'function' ? new ResizeObserver(schedule) : null;
  if (resize) resize.observe(lab.view);
  else on(window, 'resize', schedule);
  return {
    getState() { return { ...state }; },
    destroy() {
      if (destroyed) return;
      destroyed = true;
      cancelAnimationFrame(frame);
      resize?.disconnect();
      lifecycle.abort();
      lab.root.remove();
    },
  };
}

export function mountPlane(container, lang = 'ru', initial = {}) {
  return mount(container, lang, initial, 'plane');
}

export function mountCovariance(container, lang = 'ru', initial = {}) {
  return mount(container, lang, initial, 'covariance');
}
