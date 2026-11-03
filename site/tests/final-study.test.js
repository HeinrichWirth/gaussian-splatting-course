import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {photoStudy} from '../dist/compression-math.js';
import {decodeCapture} from '../dist/reconstruction-math.js';

const readJson = path => JSON.parse(readFileSync(new URL(path, import.meta.url), 'utf8'));

test('saved Python study replays in the browser with the same bytes, quality and held-out image', () => {
  const reference = readJson('../../data/codec-study/reference.json');
  const settings = readJson('../../data/codec-study/settings.json');
  const assets = '../dist/assets/reconstruction/';
  const scene = readJson(assets + 'photo-reference.json');
  const {frames} = decodeCapture(readJson(assets + 'photographs-80.json'));
  const replay = photoStudy(scene, frames, settings);
  assert.deepEqual(replay.settings, reference.settings);
  assert.equal(replay.selected, reference.selected);
  assert.equal(replay.validationView, reference.validationView);
  assert.equal(replay.test.view, 'view-04.jpg');
  assert.ok(Math.abs(replay.test.mse - reference.test.mse) < 1e-8);
  assert.equal(replay.candidates.length, reference.candidates.length);
  for (const [index, row] of replay.candidates.entries()) {
    const expected = reference.candidates[index];
    assert.deepEqual([row.name, row.bytes, row.splats], [expected.name, expected.bytes, expected.splats]);
    assert.ok(Math.abs(row.validationPsnr - expected.validationPsnr) < 1e-5);
  }
});
