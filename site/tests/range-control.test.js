import test from 'node:test';
import assert from 'node:assert/strict';
import {number} from '../dist/labs/reconstruction.js';

test('fractional range values are assigned after bounds and the picture follows normalization', () => {
  const inputs = [];
  const writes = [];
  let current = .375;
  globalThis.document = {
    createElement(tag) {
      const node = {
        append() {}, setAttribute() {},
        addEventListener(name, fn) { this[name] = fn; },
        set value(v) {
          if (this.type === 'range') writes.push([this.min, this.max, this.step, v]);
          this.textValue = String(v);
        },
        get value() { return this.textValue; },
        get valueAsNumber() { return Number(this.value); },
      };
      if (tag === 'input') inputs.push(node);
      return node;
    },
  };
  try {
    number({controls: {append() {}}}, 'Time', current, 0, 1, .005, v => { current = v; });
    assert.deepEqual(writes[0], [0, 1, .005, .375]);
    const [field, slider] = inputs;
    field.value = .379;
    field.input();
    assert.equal(current, .379);
    slider.value = .38; // The browser snapped the range to its nearest tick.
    field.blur();
    assert.equal(field.valueAsNumber, .38);
    assert.equal(current, .38);
  } finally {
    delete globalThis.document;
  }
});
