import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {experiments} from '../dist/experiments.js';
import {blocks, names, descriptions, lessonNumbers, blockForLesson, parseRoute} from '../dist/curriculum.js';

test('published lessons have a block, both languages, and working deep routes', () => {
  assert.deepEqual(lessonNumbers, Array.from({length: 30}, (_, index) => index + 1));
  for (const lang of ['ru','en']) {
    assert.equal(names[lang].length, lessonNumbers.length);
    assert.equal(descriptions[lang].length, lessonNumbers.length);
    for (const number of lessonNumbers) {
      assert.deepEqual(parseRoute(`#/${lang}/${number}`), {language: lang, number});
      assert.deepEqual(parseRoute(`#/${lang}/${number}?s=4`), {language: lang, number});
      assert.ok(blockForLesson(number).lessons.includes(number));
    }
  }
  assert.deepEqual(blocks.map(block => block.lessons.length), [5,4,6,6,5,4]);
});

test('a future lesson number cannot silently open a different lesson', () => {
  for (const hash of ['#/ru/31','#/en/99','#/ru/6oops']) assert.equal(parseRoute(hash).number,0);
});

test('every bilingual notebook opens its registered browser experiment', () => {
  const content = ['ru', 'en'].map(lang => {
    const url = new URL(`../dist/content/${lang}.json`, import.meta.url);
    return JSON.parse(readFileSync(url, 'utf8'));
  });
  const names = content.map(lessons => lessons.map(lesson => {
    const labs = lesson.cells.filter(cell => cell.type === 'lab');
    for (const lab of labs) assert.equal(typeof experiments[lab.name]?.[0], 'function', lab.name);
    return labs.map(lab => lab.name);
  }));
  assert.deepEqual(names[0], names[1]);
  assert.equal(new Set(names[0].flat()).size, Object.keys(experiments).length);
});
