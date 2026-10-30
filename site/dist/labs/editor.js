import { compositeAt } from '../math.js';
import { el, makeLab } from './shared.js';

const FORMAT = 'gaussian-splatting-course-2d';
const LIMIT = 128;
const GRID = 161;
const COPY = value => JSON.parse(JSON.stringify(value));
let sequence = 0;

const STYLE = `
.gs-editor{--ge-ink:#23323e;--ge-muted:#5c707a;--ge-border:#d9e3e7;--ge-accent:#087f83}
.gs-editor *{box-sizing:border-box}
.gs-editor .lab-body{align-items:start}
.gs-editor .lab-view,.gs-editor .lab-controls,.gs-editor .lab-readout,.gs-editor .lab-footer{min-width:0}
.gs-editor .lab-controls{display:block}
.gs-editor .lab-readout{display:block}
.gs-editor .ge-toolbar,.gs-editor .ge-actions{display:flex;flex-wrap:wrap;gap:8px;align-items:end}
.gs-editor .ge-toolbar{margin:0 0 12px}
.gs-editor .ge-toolbar label{flex:1;min-width:140px}
.gs-editor label,.gs-editor legend{font-size:13px;color:var(--ge-ink)}
.gs-editor .ge-label{display:block;margin-bottom:5px;font-weight:600}
.gs-editor select,.gs-editor input[type=number],.gs-editor textarea{font:inherit;color:var(--ge-ink);background:#fff;border:1px solid var(--ge-border);border-radius:7px}
.gs-editor select{width:100%;padding:8px;max-width:100%;text-overflow:ellipsis}
.gs-editor select[size]{padding:4px;min-height:106px}
.gs-editor select[size] option{padding:5px 7px;white-space:normal;overflow-wrap:anywhere}
.gs-editor button{font:inherit;font-size:12px;line-height:1.4;cursor:pointer;color:var(--ge-ink);background:#fff;border:1px solid var(--ge-border);border-radius:7px;padding:8px 10px}
.gs-editor button:hover:not(:disabled){border-color:var(--ge-accent);background:#edf8f7}
.gs-editor button:disabled,.gs-editor input:disabled,.gs-editor select:disabled{opacity:.45;cursor:default}
.gs-editor button.ge-primary{background:var(--ge-accent);color:#fff;border-color:var(--ge-accent)}
.gs-editor :focus-visible{outline:3px solid #57b8bf;outline-offset:3px}
.gs-editor fieldset{border:0;padding:0;margin:16px 0;min-width:0}
.gs-editor legend{font-weight:700;padding:0;margin-bottom:8px}
.gs-editor .ge-small{font-size:12px;line-height:1.55;color:var(--ge-muted);margin:8px 0}
.gs-editor .ge-range{display:block;margin:9px 0}
.gs-editor .ge-range-row{display:grid;grid-template-columns:minmax(60px,1fr) 82px;gap:8px;align-items:center}
.gs-editor input[type=range]{width:100%;min-width:0;accent-color:var(--ge-accent);margin:0}
.gs-editor input[type=number]{width:100%;min-width:0;padding:5px 4px;font-size:12px;font-variant-numeric:tabular-nums}
.gs-editor input[aria-invalid=true]{border-color:#b43030;outline-color:#b43030}
.gs-editor input[type=color]{border:1px solid var(--ge-border);background:#fff;border-radius:7px;width:42px;height:34px;padding:3px;cursor:pointer}
.gs-editor .ge-color-row{display:grid;grid-template-columns:42px repeat(3,minmax(0,1fr));gap:6px;align-items:end}
.gs-editor .ge-color-row label{display:block;min-width:0;font-size:11px}
.gs-editor .ge-color-row .ge-label{margin:0 0 2px;font-weight:400}
.gs-editor .ge-probe-controls{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:0 14px}
.gs-editor .ge-canvas{display:block;width:100%;height:auto;max-width:680px;margin:0 auto;background:white;touch-action:manipulation;border:1px solid var(--ge-border);border-radius:9px}
.gs-editor .ge-pixel{display:flex;gap:12px;align-items:center;border:1px solid var(--ge-border);background:#f6fafb;border-radius:8px;padding:12px;margin:10px 0}
.gs-editor .ge-swatch{width:52px;height:44px;flex-shrink:0;border:1px solid #a8b7be;border-radius:6px}
.gs-editor .ge-pixel-value{font-size:15px;font-weight:700;font-variant-numeric:tabular-nums;overflow-wrap:anywhere;color:var(--ge-ink)}
.gs-editor .ge-pixel-position{font-size:12px;color:var(--ge-muted);margin-top:5px}
.gs-editor .ge-toggle{display:flex;gap:7px;align-items:center;font-size:12px;margin:10px 0}
.gs-editor .ge-toggle input{accent-color:var(--ge-accent)}
.gs-editor .ge-table-wrap{overflow:auto;max-height:400px;border:1px solid var(--ge-border);border-radius:8px;margin:12px 0}
.gs-editor .ge-trace{border-collapse:collapse;width:100%;font-size:12px;font-variant-numeric:tabular-nums;color:var(--ge-ink);margin:0}
.gs-editor .ge-trace caption{text-align:left;background:#edf4f5;padding:10px;font-weight:600;min-width:690px}
.gs-editor .ge-trace th,.gs-editor .ge-trace td{padding:8px 10px;text-align:right;border-bottom:1px solid var(--ge-border);white-space:nowrap}
.gs-editor .ge-trace thead th{background:#edf4f5;position:sticky;top:0;z-index:1;font-size:11px}
.gs-editor .ge-trace th:first-child,.gs-editor .ge-trace td:first-child{text-align:left;white-space:normal;min-width:115px;max-width:200px;overflow-wrap:anywhere}
.gs-editor .ge-trace tbody tr:nth-child(even){background:#f8fafb}
.gs-editor .ge-trace tbody tr.ge-selected{background:#e8f4f2}
.gs-editor .ge-equation{font-size:13px;color:var(--ge-ink);line-height:1.65;overflow-wrap:anywhere;margin:7px 0}
.gs-editor .ge-json{border-top:1px solid var(--ge-border);padding-top:12px;margin-top:12px}
.gs-editor .ge-json summary{cursor:pointer;font-size:14px;font-weight:600;color:var(--ge-ink)}
.gs-editor .ge-json textarea{width:100%;height:170px;padding:10px;resize:vertical;font-family:ui-monospace,monospace;font-size:12px;line-height:1.5;margin:10px 0}
.gs-editor .ge-status{font-size:12px;line-height:1.5;color:var(--ge-muted);min-height:1.5em;margin:8px 0}
.gs-editor .ge-status[data-error=true]{color:#a52b2b}
@media(max-width:620px){.gs-editor .ge-probe-controls{grid-template-columns:1fr}.gs-editor .ge-actions button{flex:1}.gs-editor .ge-pixel-value{font-size:13px}.gs-editor .ge-trace th,.gs-editor .ge-trace td{padding:7px}.gs-editor .ge-range-row{grid-template-columns:minmax(70px,1fr) 88px}}
`;

function presetScene(kind, t) {
  const layer = (name, center, scales, angle_deg, color, opacity) => ({name, center, scales, angle_deg, color, opacity});
  if (kind === 'blank') return {layers: [], background: [1, 1, 1]};
  if (kind !== 'flower') return {background: [1, 1, 1], layers: [
    layer(t('Красное', 'Red'), [0, 0], [1.5, 1], 0, [1, 0, 0], .5),
    layer(t('Синее', 'Blue'), [0, 0], [1.5, 1], 0, [0, 0, 1], .5),
  ]};
  const layers = [layer(t('Середина', 'Center'), [0, .65], [.43, .43], 0, [1, .65, .06], .98)];
  [0, 60, 120, 180, 240, 300].forEach((angle, i) => {
    const radians = angle * Math.PI / 180;
    layers.push(layer(t(`Лепесток ${i + 1}`, `Petal ${i + 1}`),
      [.95 * Math.cos(radians), .65 + .95 * Math.sin(radians)], [.75, .38],
      (angle + 180) % 360 - 180, [.92, .15 + .045 * i, .38 + .035 * i], .92));
  });
  layers.push(
    layer(t('Лист слева', 'Left leaf'), [-.60, -1.35], [.75, .24], -28, [.15, .55, .22], .92),
    layer(t('Лист справа', 'Right leaf'), [.58, -1.8], [.7, .24], 30, [.12, .48, .2], .92),
    layer(t('Стебель', 'Stem'), [0, -1.2], [.2, 1.45], 0, [.12, .45, .20], .95),
  );
  return {layers, background: [1, 1, 1]};
}

function validateScene(scene, t) {
  const fail = (ru, en) => { throw new Error(t(ru, en)); };
  const numeric = (x, low, high) => typeof x === 'number' && Number.isFinite(x) && x >= low && x <= high;
  const tuple = (v, n, low, high) => Array.isArray(v) && v.length === n && v.every(x => numeric(x, low, high));
  if (!scene || typeof scene !== 'object' || Array.isArray(scene) || scene.format !== FORMAT || scene.version !== 1) {
    fail('Формат сцены не распознан. Редактор использует gaussian-splatting-course-2d, версия 1.',
      'Unrecognized scene format. The editor uses gaussian-splatting-course-2d, version 1.');
  }
  if (!Array.isArray(scene.layers) || scene.layers.length > LIMIT) {
    fail('В сцене допустимо от 0 до 128 пятен.', 'A scene can contain from 0 to 128 spots.');
  }
  if (!tuple(scene.background, 3, 0, 1)) fail('Фон содержит три числа RGB от 0 до 1.', 'The background contains three RGB numbers from 0 to 1.');
  const keys = ['angle_deg', 'center', 'color', 'name', 'opacity', 'scales'];
  scene.layers.forEach((layer, index) => {
    const prefix = t(`Пятно ${index + 1}: `, `Spot ${index + 1}: `);
    if (!layer || typeof layer !== 'object' || Array.isArray(layer) || Object.keys(layer).sort().join(',') !== keys.join(',')) {
      throw new Error(prefix + t('ожидаются name, center, scales, angle_deg, color, opacity.', 'expected name, center, scales, angle_deg, color, opacity.'));
    }
    if (typeof layer.name !== 'string' || !layer.name.trim() || Array.from(layer.name).length > 80) {
      throw new Error(prefix + t('имя содержит от 1 до 80 символов.', 'the name contains from 1 to 80 characters.'));
    }
    if (!tuple(layer.center, 2, -4, 4) || !tuple(layer.scales, 2, .2, 3) || !numeric(layer.angle_deg, -180, 180)) {
      throw new Error(prefix + t('центры от −4 до 4, масштабы от 0.2 до 3, угол от −180° до 180°.', 'centers range from −4 to 4, scales from 0.2 to 3, angles from −180° to 180°.'));
    }
    if (!tuple(layer.color, 3, 0, 1) || !numeric(layer.opacity, 0, 1)) {
      throw new Error(prefix + t('RGB и непрозрачность лежат от 0 до 1.', 'RGB and opacity range from 0 to 1.'));
    }
  });
  // Copy only canonical fields, retaining all numerical precision.
  return COPY({format: FORMAT, version: 1, background: scene.background, layers: scene.layers});
}

function hex(color) { return '#' + color.map(v => Math.round(v * 255).toString(16).padStart(2, '0')).join(''); }
function rgb(value) { return [1, 3, 5].map(i => parseInt(value.slice(i, i + 2), 16) / 255); }
function number(value, digits = 6) {
  if (value === 0) return '0';
  if (Math.abs(value) < 1e-5) return value.toExponential(3);
  return Number(value.toFixed(digits)).toString();
}
function colorText(value) { return '(' + value.map(v => number(v)).join(', ') + ')'; }

/** A local, front-to-back Gaussian editor. getState() returns an independent copy. */
export function mountEditor(container, lang = 'ru', initial = {}) {
  const t = (ru, en) => lang === 'en' ? en : ru;
  const id = `gaussian-editor-${++sequence}`;
  const events = new AbortController();
  const on = (node, type, handler) => node.addEventListener(type, handler, {signal: events.signal});
  let destroyed = false, frame = 0, imageDirty = true, importGeneration = 0;
  let scene = presetScene(initial.preset, t);
  let initialError = '';
  if (initial.scene || initial.layers !== undefined || initial.background !== undefined) {
    try {
      const value = initial.scene || {format: FORMAT, version: 1, ...initial};
      scene = validateScene(value, t);
    } catch (error) { initialError = error.message; }
  }
  let selected = scene.layers.length ? Math.max(0, Math.min(scene.layers.length - 1, Number.isInteger(initial.selected) ? initial.selected : 0)) : -1;
  const validProbe = v => typeof v === 'number' && Number.isFinite(v) && Math.abs(v) <= 4;
  const startingProbe = Array.isArray(initial.probe) ? initial.probe : [0, 0];
  let probe = startingProbe.map(v => validProbe(v) ? Math.round(v * 20) / 20 : 0).slice(0, 2);
  if (probe.length !== 2) probe = [0, 0];
  let outline = initial.outline !== false;
  const lab = makeLab(container, {
    lang,
    title: t('Редактор гауссиан', 'Gaussian editor'),
    subtitle: t('Форма, цвет и порядок слоёв — в одной картинке. У каждого пикселя есть свой расчёт.',
      'Shape, color, and layer order in one image. Every pixel has its own calculation.'),
    tag: t('ОПЫТ', 'EXPERIMENT'),
  });
  lab.root.classList.add('gs-editor');
  lab.root.dataset.editor = id;
  lab.root.append(el('style', '', STYLE));

  function button(parent, text, action, key, primary = false) {
    const result = el('button', primary ? 'ge-primary' : '', text);
    result.type = 'button';
    result.dataset.action = key;
    on(result, 'click', action);
    parent.append(result);
    return result;
  }
  function label(parent, caption) {
    const row = el('label');
    row.append(el('span', 'ge-label', caption));
    parent.append(row);
    return row;
  }
  function fieldset(parent, caption) {
    const group = el('fieldset');
    group.append(el('legend', '', caption));
    parent.append(group);
    return group;
  }
  function range(parent, key, caption, low, high, step, read, write, isProbe = false) {
    const wrapper = el('div', 'ge-range');
    const captionNode = el('label', 'ge-label', caption);
    const row = el('div', 'ge-range-row');
    const slider = el('input');
    Object.assign(slider, {type: 'range', min: low, max: high, step, id: `${id}-${key}`, value: read()});
    slider.dataset.control = key;
    captionNode.htmlFor = slider.id;
    const input = el('input');
    Object.assign(input, {type: 'number', min: low, max: high, step: 'any', value: read()});
    input.dataset.control = `${key}-number`;
    input.setAttribute('aria-label', t(`${caption}, число`, `${caption}, number`));
    function apply(value, source) {
      if (!Number.isFinite(value) || value < low || value > high || source.value.trim() === '') {
        source.setAttribute('aria-invalid', 'true');
        setStatus(t(`${caption}: диапазон от ${low} до ${high}.`, `${caption}: range ${low} to ${high}.`), true);
        return;
      }
      if (isProbe) value = Math.round(value * 20) / 20;
      source.removeAttribute('aria-invalid');
      write(value);
      slider.value = value;
      if (source !== input || isProbe) input.value = value;
      setStatus('');
      schedule(!isProbe);
    }
    on(slider, 'input', () => apply(Number(slider.value), slider));
    on(input, 'input', () => apply(input.valueAsNumber, input));
    row.append(slider, input);
    wrapper.append(captionNode, row);
    parent.append(wrapper);
    return {
      sync() { slider.value = read(); input.value = read(); input.removeAttribute('aria-invalid'); },
      disable(value) { slider.disabled = input.disabled = value; },
    };
  }
  function colorControls(parent, key, caption, read, write) {
    const group = fieldset(parent, caption);
    const row = el('div', 'ge-color-row');
    const picker = el('input');
    picker.type = 'color';
    picker.dataset.control = key;
    picker.setAttribute('aria-label', caption);
    row.append(picker);
    const numbers = ['R', 'G', 'B'].map((channel, i) => {
      const wrap = label(row, channel);
      const input = el('input');
      Object.assign(input, {type: 'number', min: 0, max: 1, step: 'any'});
      input.dataset.control = `${key}-${channel.toLowerCase()}`;
      input.setAttribute('aria-label', `${caption}, ${channel}`);
      wrap.append(input);
      on(input, 'input', () => {
        const value = input.valueAsNumber;
        if (!Number.isFinite(value) || value < 0 || value > 1) {
          input.setAttribute('aria-invalid', 'true');
          setStatus(t(`${caption}: RGB от 0 до 1.`, `${caption}: RGB from 0 to 1.`), true);
          return;
        }
        input.removeAttribute('aria-invalid');
        const next = [...read()];
        next[i] = value;
        write(next);
        picker.value = hex(next);
        setStatus('');
        schedule(true);
      });
      return input;
    });
    on(picker, 'input', () => {
      write(rgb(picker.value));
      numbers.forEach((input, i) => { input.value = read()[i]; input.removeAttribute('aria-invalid'); });
      setStatus('');
      schedule(true);
    });
    group.append(row);
    return {
      sync() {
        picker.value = hex(read());
        numbers.forEach((input, i) => { input.value = read()[i]; input.removeAttribute('aria-invalid'); });
      },
      disable(value) { picker.disabled = value; numbers.forEach(input => { input.disabled = value; }); },
    };
  }

  const toolbar = el('div', 'ge-toolbar');
  lab.view.append(toolbar);
  const presetLabel = label(toolbar, t('Пример', 'Example'));
  const preset = el('select');
  preset.dataset.control = 'preset';
  for (const [value, caption] of [['redblue', t('Красное и синее', 'Red and blue')], ['flower', t('Цветок', 'Flower')], ['blank', t('Чистый холст', 'Blank canvas')]]) {
    const option = el('option', '', caption); option.value = value; preset.append(option);
  }
  preset.value = ['flower', 'blank'].includes(initial.preset) ? initial.preset : 'redblue';
  presetLabel.append(preset);
  button(toolbar, t('Загрузить пример', 'Load example'), () => {
    importGeneration++;
    replaceScene(presetScene(preset.value, t));
    setStatus(t('Пример загружен.', 'Example loaded.'));
  }, 'load-preset');
  button(toolbar, t('Сбросить всё', 'Reset all'), () => {
    importGeneration++;
    preset.value = 'redblue';
    probe = [0, 0]; outline = true;
    json.value = ''; file.value = '';
    replaceScene(presetScene('redblue', t));
    setStatus(t('Исходные два пятна восстановлены.', 'The original two spots are restored.'));
  }, 'reset');

  const canvas = el('canvas', 'ge-canvas');
  canvas.width = canvas.height = 640;
  canvas.tabIndex = 0;
  canvas.setAttribute('role', 'img');
  canvas.setAttribute('aria-describedby', `${id}-canvas-note`);
  canvas.setAttribute('aria-keyshortcuts', 'ArrowLeft ArrowRight ArrowUp ArrowDown');
  canvas.dataset.control = 'canvas';
  lab.view.append(canvas);
  const context = canvas.getContext('2d');
  const raster = document.createElement('canvas');
  raster.width = raster.height = GRID;
  const rasterContext = raster.getContext('2d');
  const image = rasterContext.createImageData(GRID, GRID);
  const canvasNote = el('p', 'ge-small', t('Клик на картинке или стрелки при фокусе на ней меняют проверяемый пиксель. Пятна остаются на месте.',
    'A click on the image, or arrow keys while it has focus, changes the probed pixel. The spots stay in place.'));
  canvasNote.id = `${id}-canvas-note`;
  lab.view.append(canvasNote);

  const pixelPanel = el('div', 'ge-pixel');
  const swatch = el('span', 'ge-swatch');
  swatch.setAttribute('aria-hidden', 'true');
  const pixelDetails = el('div');
  const pixelColor = el('div', 'ge-pixel-value');
  pixelColor.dataset.readout = 'pixel-color';
  const pixelPosition = el('div', 'ge-pixel-position');
  pixelDetails.append(pixelColor, pixelPosition); pixelPanel.append(swatch, pixelDetails); lab.view.append(pixelPanel);
  const probeControls = el('div', 'ge-probe-controls');
  lab.view.append(probeControls);
  const probeInputs = [0, 1].map(i => range(probeControls, i ? 'probe-y' : 'probe-x',
    i ? t('Пиксель y', 'Pixel y') : t('Пиксель x', 'Pixel x'), -4, 4, .05,
    () => probe[i], value => { probe[i] = value; }, true));
  const background = colorControls(lab.view, 'background', t('Цвет фона', 'Background color'),
    () => scene.background, value => { scene.background = value; });

  const layersGroup = fieldset(lab.controls, t('Слои · первый спереди', 'Layers · first is in front'));
  const select = el('select');
  select.size = 4;
  select.dataset.control = 'selected';
  select.setAttribute('aria-label', t('Выбранное пятно', 'Selected spot'));
  layersGroup.append(select);
  const layerCount = el('p', 'ge-small');
  layersGroup.append(layerCount);
  const orderActions = el('div', 'ge-actions'); layersGroup.append(orderActions);
  const front = button(orderActions, t('↑ На слой ближе', '↑ One layer forward'), () => move(-1), 'front');
  const back = button(orderActions, t('↓ На слой дальше', '↓ One layer back'), () => move(1), 'back');
  const layerActions = el('div', 'ge-actions'); layerActions.style.marginTop = '8px'; layersGroup.append(layerActions);
  const add = button(layerActions, t('+ Добавить пятно', '+ Add spot'), () => {
    if (scene.layers.length >= LIMIT) return;
    const names = new Set(scene.layers.map(layer => layer.name));
    let n = scene.layers.length + 1;
    while (names.has(t(`Пятно ${n}`, `Spot ${n}`))) n++;
    scene.layers.unshift({name: t(`Пятно ${n}`, `Spot ${n}`), center: [0, 0], scales: [.7, .7], angle_deg: 0, color: [1, .65, .1], opacity: .7});
    selected = 0; sync(); schedule(true);
    setStatus(t('Новое пятно добавлено спереди.', 'A new spot was added at the front.'));
  }, 'add', true);
  const remove = button(layerActions, t('Удалить пятно', 'Remove spot'), () => {
    if (selected < 0) return;
    scene.layers.splice(selected, 1);
    selected = Math.min(selected, scene.layers.length - 1);
    sync(); schedule(true);
    setStatus(t('Пятно удалено.', 'Spot removed.'));
  }, 'remove');
  on(select, 'change', () => { selected = Number(select.value); sync(); schedule(false); });
  const layerGroup = fieldset(lab.controls, t('Параметры выбранного пятна', 'Selected spot parameters'));
  const current = () => scene.layers[selected];
  const settings = [
    ['x', t('Центр x', 'Center x'), -4, 4, .05, 'center', 0],
    ['y', t('Центр y', 'Center y'), -4, 4, .05, 'center', 1],
    ['sx', t('Масштаб s₁', 'Scale s₁'), .2, 3, .05, 'scales', 0],
    ['sy', t('Масштаб s₂', 'Scale s₂'), .2, 3, .05, 'scales', 1],
    ['angle', t('Поворот, °', 'Rotation, °'), -180, 180, 1, 'angle_deg'],
    ['opacity', t('Непрозрачность o', 'Opacity o'), 0, 1, .05, 'opacity'],
  ];
  const fields = settings.map(([key, caption, low, high, step, property, component]) => range(layerGroup, key, caption, low, high, step,
    () => current() ? (component === undefined ? current()[property] : current()[property][component]) : low,
    value => {
      if (!current()) return;
      // Only the edited property changes. Imported RGB and other values stay exact.
      if (component === undefined) current()[property] = value;
      else current()[property][component] = value;
    }));
  const color = colorControls(lab.controls, 'color', t('Цвет пятна · RGB', 'Spot color · RGB'),
    () => current()?.color || [0, 0, 0], value => { if (current()) current().color = value; });
  const toggleLabel = el('label', 'ge-toggle');
  const toggle = el('input'); toggle.type = 'checkbox'; toggle.checked = outline; toggle.dataset.control = 'outline';
  toggleLabel.append(toggle, document.createTextNode(t('Контур выбранного пятна', 'Selected spot contour')));
  lab.controls.append(toggleLabel);
  on(toggle, 'change', () => { outline = toggle.checked; schedule(false); });
  lab.controls.append(el('p', 'ge-small', t('Пунктир — уровень G ≈ 0.607, а не граница пятна. Масштабы от 0.2 до 3; центры от −4 до 4.',
    'The dashed contour is G ≈ 0.607, not the edge of the spot. Scales range from 0.2 to 3; centers from −4 to 4.')));

  lab.readout.append(el('p', 'ge-equation', 'α = oG     ·     C ← C + Tαc     ·     T ← T(1 − α)     ·     Cfinal = C + TB'));
  const tableWrap = el('div', 'ge-table-wrap');
  tableWrap.tabIndex = 0;
  tableWrap.setAttribute('role', 'region');
  tableWrap.setAttribute('aria-label', t('Пошаговый расчёт пикселя', 'Step-by-step pixel calculation'));
  const table = el('table', 'ge-trace');
  const caption = el('caption'); table.append(caption);
  const thead = el('thead'), headerRow = el('tr');
  [t('Слой: спереди → назад', 'Layer: front → back'), 'G', 'o', 'α = oG', t('T до', 'T before'), 'Tα', t('Вклад RGB', 'Added RGB'), t('T после', 'T after')].forEach(value => {
    const th = el('th', '', value); th.scope = 'col'; headerRow.append(th);
  });
  thead.append(headerRow); table.append(thead);
  const tbody = el('tbody'); table.append(tbody); tableWrap.append(table); lab.readout.append(tableWrap);
  lab.readout.append(el('p', 'ge-small', t('T — оставшаяся прозрачность перед следующим слоем. В конце вся оставшаяся доля достаётся фону. RGB смешиваются как числа без отдельной гамма-коррекции.',
    'T is the remaining transmittance before the next layer. The background receives the final remainder. RGB values are mixed as numbers without a separate gamma-correction step.')));

  const storage = el('details', 'ge-json');
  storage.append(el('summary', '', t('Сохранение сцены · JSON', 'Scene storage · JSON')));
  const storageHint = el('p', 'ge-small', t('Файл совместим с редактором из ноутбука. Сохраняются пятна, их порядок и фон; положение крестика в файл не входит. Импорт заменяет сцену только после полной проверки.',
    'The file is compatible with the notebook editor. Spots, their order, and the background are saved; the cross position is not included. Import replaces the scene only after complete validation.'));
  storage.append(storageHint);
  const storageActions = el('div', 'ge-actions'); storage.append(storageActions);
  const json = el('textarea');
  json.dataset.control = 'json';
  json.spellcheck = false;
  json.maxLength = 1000000;
  json.setAttribute('aria-label', t('Параметры сцены в JSON', 'Scene parameters as JSON'));
  json.placeholder = t('Здесь появляется экспорт сцены. Это же поле принимает сохранённый JSON.', 'Exported scene JSON appears here. The same field accepts a saved scene.');
  button(storageActions, t('Скачать JSON', 'Download JSON'), exportScene, 'export');
  button(storageActions, t('Открыть JSON-файл', 'Open JSON file'), () => file.click(), 'open-file');
  button(storageActions, t('Применить JSON из поля', 'Apply JSON from field'), () => { importGeneration++; importText(json.value); }, 'import');
  const file = el('input'); file.type = 'file'; file.accept = '.json,application/json'; file.hidden = true;
  file.dataset.control = 'file';
  on(file, 'change', async () => {
    const selectedFile = file.files?.[0];
    if (!selectedFile) return;
    const generation = ++importGeneration;
    if (selectedFile.size > 1000000) {
      setStatus(t('Сцена не загружена: файл больше 1 МБ.', 'Scene not loaded: the file exceeds 1 MB.'), true);
      file.value = ''; return;
    }
    try {
      const value = await selectedFile.text();
      if (!destroyed && generation === importGeneration && importText(value)) json.value = value;
    } catch {
      if (!destroyed && generation === importGeneration) setStatus(t('Файл не удалось прочитать. Текущая сцена сохранена.', 'The file could not be read. The current scene is unchanged.'), true);
    }
    file.value = '';
  });
  storage.append(json, file);
  lab.footer.append(storage);
  const status = el('p', 'ge-status');
  status.setAttribute('role', 'status'); status.setAttribute('aria-live', 'polite');
  lab.footer.append(status);
  function setStatus(message, error = false) { status.textContent = message; status.dataset.error = String(error); }

  function canonical() { return {format: FORMAT, version: 1, background: scene.background, layers: scene.layers}; }
  function exportScene() {
    const text = JSON.stringify(canonical(), null, 2);
    json.value = text;
    const url = URL.createObjectURL(new Blob([text], {type: 'application/json'}));
    const anchor = el('a'); anchor.href = url; anchor.download = 'gaussian-scene.json'; anchor.hidden = true;
    lab.root.append(anchor); anchor.click(); anchor.remove();
    // A brief delay allows the browser to start the download before revocation.
    const timer = setTimeout(() => { URL.revokeObjectURL(url); downloads.delete(url); }, 1000);
    downloads.set(url, timer);
    setStatus(t('JSON подготовлен для скачивания и сохранён в текстовом поле.', 'JSON is ready to download and is also in the text field.'));
  }
  const downloads = new Map();
  function importText(value) {
    try {
      if (typeof value !== 'string' || value.length > 1000000) throw new Error(t('Допустимо до 1 млн символов JSON.', 'JSON is limited to one million characters.'));
      let parsed;
      try { parsed = JSON.parse(value); }
      catch { throw new Error(t('Текст не является корректным JSON.', 'The text is not valid JSON.')); }
      const validated = validateScene(parsed, t);
      replaceScene(validated);
      setStatus(t('Сцена загружена.', 'Scene loaded.'));
      return true;
    } catch (error) {
      setStatus(t('Сцена не загружена. ', 'Scene not loaded. ') + error.message + t(' Текущая сцена сохранена.', ' The current scene is unchanged.'), true);
      return false;
    }
  }
  function replaceScene(next) { scene = COPY(next); selected = scene.layers.length ? 0 : -1; sync(); schedule(true); }
  function move(direction) {
    const target = selected + direction;
    if (selected < 0 || target < 0 || target >= scene.layers.length) return;
    [scene.layers[selected], scene.layers[target]] = [scene.layers[target], scene.layers[selected]];
    selected = target; sync(); schedule(true);
    setStatus(t('Порядок слоёв изменён.', 'Layer order changed.'));
  }
  function sync() {
    select.replaceChildren();
    scene.layers.forEach((layer, i) => { const option = el('option', '', `${i + 1}. ${layer.name}`); option.value = i; select.append(option); });
    if (!scene.layers.length) { const option = el('option', '', t('Только фон', 'Background only')); option.value = -1; select.append(option); }
    select.value = selected;
    const empty = selected < 0;
    select.disabled = remove.disabled = empty;
    front.disabled = empty || selected === 0;
    back.disabled = empty || selected === scene.layers.length - 1;
    add.disabled = scene.layers.length >= LIMIT;
    layerCount.textContent = t(`${scene.layers.length} из ${LIMIT} пятен · первый слой спереди`, `${scene.layers.length} of ${LIMIT} spots · first layer is in front`);
    fields.forEach(input => { input.sync(); input.disable(empty); });
    color.sync(); color.disable(empty); background.sync();
    probeInputs.forEach(input => input.sync());
    toggle.checked = outline;
  }
  function schedule(rebuild = false) {
    if (destroyed) return;
    imageDirty ||= rebuild;
    if (!frame) frame = requestAnimationFrame(() => { frame = 0; draw(); });
  }
  function renderImage() {
    // This is compositeAt's front-to-back calculation with constant transforms
    // prepared once per layer, rather than recomputed for each of 25,921 pixels.
    const layers = scene.layers.map(layer => {
      const angle = layer.angle_deg * Math.PI / 180;
      return {...layer, cosine: Math.cos(angle), sine: Math.sin(angle)};
    });
    let index = 0;
    for (let row = 0; row < GRID; row++) {
      const y = 4 - row * 8 / (GRID - 1);
      for (let column = 0; column < GRID; column++) {
        const x = -4 + column * 8 / (GRID - 1);
        let r = 0, g = 0, b = 0, remaining = 1;
        for (const layer of layers) {
          const dx = x - layer.center[0], dy = y - layer.center[1];
          const u = (layer.cosine * dx + layer.sine * dy) / layer.scales[0];
          const v = (-layer.sine * dx + layer.cosine * dy) / layer.scales[1];
          const alpha = layer.opacity * Math.exp(-.5 * (u * u + v * v));
          const share = remaining * alpha;
          r += share * layer.color[0]; g += share * layer.color[1]; b += share * layer.color[2];
          remaining *= 1 - alpha;
        }
        image.data[index++] = Math.round(255 * (r + remaining * scene.background[0]));
        image.data[index++] = Math.round(255 * (g + remaining * scene.background[1]));
        image.data[index++] = Math.round(255 * (b + remaining * scene.background[2]));
        image.data[index++] = 255;
      }
    }
    rasterContext.putImageData(image, 0, 0);
    imageDirty = false;
  }
  const PLOT = {left: 46, top: 24, size: 552};
  const px = x => PLOT.left + (x + 4) / 8 * PLOT.size;
  const py = y => PLOT.top + (4 - y) / 8 * PLOT.size;
  function draw() {
    if (destroyed) return;
    if (imageDirty) renderImage();
    context.clearRect(0, 0, 640, 640);
    context.fillStyle = '#fff'; context.fillRect(0, 0, 640, 640);
    context.save(); context.beginPath(); context.rect(PLOT.left, PLOT.top, PLOT.size, PLOT.size); context.clip();
    context.imageSmoothingEnabled = false;
    // The endpoint samples sit exactly at ±4, with half a pixel beyond each edge.
    const half = PLOT.size / (GRID - 1) / 2;
    context.drawImage(raster, PLOT.left - half, PLOT.top - half, PLOT.size + 2 * half, PLOT.size + 2 * half);
    if (outline && current()) {
      const layer = current();
      context.beginPath();
      context.ellipse(px(layer.center[0]), py(layer.center[1]), layer.scales[0] / 8 * PLOT.size, layer.scales[1] / 8 * PLOT.size, -layer.angle_deg * Math.PI / 180, 0, Math.PI * 2);
      context.strokeStyle = '#fff'; context.lineWidth = 3; context.setLineDash([7, 5]); context.stroke();
      context.strokeStyle = '#23323e'; context.lineWidth = 1.3; context.stroke(); context.setLineDash([]);
      context.beginPath(); context.arc(px(layer.center[0]), py(layer.center[1]), 3, 0, Math.PI * 2); context.fillStyle = '#23323e'; context.fill();
    }
    context.beginPath(); context.moveTo(px(probe[0]) - 9, py(probe[1])); context.lineTo(px(probe[0]) + 9, py(probe[1]));
    context.moveTo(px(probe[0]), py(probe[1]) - 9); context.lineTo(px(probe[0]), py(probe[1]) + 9);
    context.strokeStyle = '#fff'; context.lineWidth = 5; context.stroke(); context.strokeStyle = '#23323e'; context.lineWidth = 2; context.stroke();
    context.restore();
    context.strokeStyle = '#83969e'; context.lineWidth = 1; context.strokeRect(PLOT.left, PLOT.top, PLOT.size, PLOT.size);
    context.font = '13px system-ui, sans-serif'; context.fillStyle = '#526873';
    for (let tick = -4; tick <= 4; tick++) {
      context.textAlign = 'center'; context.fillText(tick, px(tick), PLOT.top + PLOT.size + 22);
      context.textAlign = 'right'; context.fillText(tick, PLOT.left - 11, py(tick) + 4);
    }
    context.textAlign = 'center'; context.fillText('x', PLOT.left + PLOT.size / 2, 624);
    context.fillText('y', 18, 14);
    const trace = compositeAt(probe[0], probe[1], scene.layers, scene.background);
    pixelColor.textContent = 'RGB = ' + colorText(trace.color);
    pixelPosition.textContent = `(x, y) = ${colorText(probe)} · ${t('Осталось фону', 'Remaining for background')} T = ${number(trace.transmittance)}`;
    swatch.style.background = `rgb(${trace.color.map(v => v * 255).join(',')})`;
    canvas.setAttribute('aria-label', t(`Изображение из ${scene.layers.length} гауссиан. Пиксель ${colorText(probe)}: RGB ${colorText(trace.color)}.`,
      `Image of ${scene.layers.length} Gaussians. Pixel ${colorText(probe)}: RGB ${colorText(trace.color)}.`));
    caption.textContent = t(`Расчёт пикселя ${colorText(probe)} · начальные C = (0, 0, 0), T = 1`, `Pixel ${colorText(probe)} · initial C = (0, 0, 0), T = 1`);
    tbody.replaceChildren();
    trace.layers.forEach((step, index) => {
      const row = el('tr', index === selected ? 'ge-selected' : '');
      const values = [`${index + 1}. ${step.name}`, number(step.weight), number(scene.layers[index].opacity), number(step.alpha), number(step.before), number(step.before * step.alpha), colorText(step.contribution), number(step.after)];
      values.forEach((value, column) => {
        const cell = el(column === 0 ? 'th' : 'td', '', value);
        if (column === 0) cell.scope = 'row';
        row.append(cell);
      });
      tbody.append(row);
    });
    const bg = el('tr');
    [t('Фон', 'Background'), '—', '1', '—', number(trace.transmittance), number(trace.transmittance), colorText(trace.backgroundContribution), '0'].forEach((value, column) => {
      const cell = el(column === 0 ? 'th' : 'td', '', value); if (!column) cell.scope = 'row'; bg.append(cell);
    });
    tbody.append(bg);
  }
  function moveProbe(x, y) {
    probe = [x, y].map(value => Math.max(-4, Math.min(4, Math.round(value * 20) / 20)));
    probeInputs.forEach(input => input.sync()); schedule(false);
  }
  on(canvas, 'pointerdown', event => {
    const box = canvas.getBoundingClientRect();
    const x = (event.clientX - box.left) / box.width * 640;
    const y = (event.clientY - box.top) / box.height * 640;
    if (x < PLOT.left || x > PLOT.left + PLOT.size || y < PLOT.top || y > PLOT.top + PLOT.size) return;
    moveProbe((x - PLOT.left) / PLOT.size * 8 - 4, 4 - (y - PLOT.top) / PLOT.size * 8);
    canvas.focus({preventScroll: true});
  });
  on(canvas, 'keydown', event => {
    const delta = {ArrowLeft: [-.05, 0], ArrowRight: [.05, 0], ArrowUp: [0, .05], ArrowDown: [0, -.05]}[event.key];
    if (delta) { event.preventDefault(); moveProbe(probe[0] + delta[0], probe[1] + delta[1]); }
  });
  sync(); draw();
  if (initialError) setStatus(t('Начальная сцена не загружена. ', 'Initial scene not loaded. ') + initialError, true);
  return {
    getState() {
      return COPY({...canonical(), selected, probe, outline, preset: preset.value,
        pixel: compositeAt(probe[0], probe[1], scene.layers, scene.background)});
    },
    destroy() {
      destroyed = true; importGeneration++;
      events.abort(); if (frame) cancelAnimationFrame(frame);
      downloads.forEach((timer, url) => { clearTimeout(timer); URL.revokeObjectURL(url); }); downloads.clear();
      lab.root.remove();
    },
  };
}
