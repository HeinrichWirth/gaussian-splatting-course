import test from 'node:test';
import assert from 'node:assert/strict';
import {blocks, names, descriptions, lessonNumbers, blockForLesson, parseRoute} from '../dist/curriculum.js';

test('published lessons have a block, both languages, and working deep routes', () => {
  assert.deepEqual(lessonNumbers, [1,2,3,4,5,6,7,8,9]);
  for (const lang of ['ru','en']) {
    assert.equal(names[lang].length, lessonNumbers.length);
    assert.equal(descriptions[lang].length, lessonNumbers.length);
    for (const number of lessonNumbers) {
      assert.deepEqual(parseRoute(`#/${lang}/${number}`), {language: lang, number});
      assert.deepEqual(parseRoute(`#/${lang}/${number}?s=4`), {language: lang, number});
      assert.ok(blockForLesson(number).lessons.includes(number));
    }
  }
  assert.deepEqual(blocks.map(block => block.lessons.length), [5,4]);
});

test('a future lesson number cannot silently open a different lesson', () => {
  for (const hash of ['#/ru/10','#/en/99','#/ru/6oops']) assert.equal(parseRoute(hash).number,0);
});
