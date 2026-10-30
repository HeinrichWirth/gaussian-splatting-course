// The same lesson numbers identify notebooks, exported text, and browser routes.
export const blocks = [
  {number: 1, lessons: [1, 2, 3, 4, 5],
    title: {ru: 'Как из гауссиан получается изображение', en: 'How Gaussians become an image'},
    short: {ru: 'Форма и цвет', en: 'Shape and color'},
    description: {ru: 'От одного колокола до картинки, собранной вручную.', en: 'From one bell curve to an image assembled by hand.'}},
  {number: 2, lessons: [6, 7, 8, 9],
    title: {ru: 'Как гауссианы учатся', en: 'How Gaussians learn'},
    short: {ru: 'Обучение', en: 'Learning'},
    description: {ru: 'Ошибка показывает расхождение, градиент задаёт направление. Пятна постепенно складываются в образец.', en: 'The loss measures the mismatch; the gradient gives a direction. The spots gradually take the shape of the target.'}},
];
export const names = {
  ru: ['Из пятен — в картинку', 'Один колокол', 'Гауссиана на плоскости', 'Поворот и матрица формы', 'Несколько пятен', 'Как измерить ошибку', 'Куда двигать параметр', 'От цепочки к обучению', 'Восстанавливаем картинку'],
  en: ['From splats to an image', 'One bell curve', 'A Gaussian on a plane', 'Rotation and the shape matrix', 'Several splats', 'Measuring the error', 'Which way to move', 'From a chain to learning', 'Reconstructing an image'],
};
export const descriptions = {
  ru: ['Сцена, камера и 192 гауссианы. Что вообще хранится внутри.', 'Центр, ширина и вес. Одна формула — пять простых действий.', 'Два направления, цветная карта и холм. От круга к эллипсу.', 'Свои оси пятна, поворот и четыре числа ковариации.', 'Цвет, прозрачность и порядок. Маленький редактор сцены.', 'Образец, результат и карта различий. От четырёх чисел до ошибки всего изображения.', 'Наклон графика, маленький шаг и слишком большой шаг. Производная на конкретных числах.', 'Каждая операция оставляет след. Правило цепочки, PyTorch и первый цикл обучения.', 'Десятки и сотня гауссиан восстанавливают изображение. Начальные параметры, детали и сохранение сцены.'],
  en: ['A scene, a camera, and 192 Gaussians. What is actually stored.', 'Center, width, and weight. One formula in five simple steps.', 'Two directions, a heatmap, and a hill. From a circle to an ellipse.', 'Local axes, rotation, and the four entries of covariance.', 'Color, opacity, and order. A small scene editor.', 'Target, prediction, and a difference map. From four numbers to a loss for the whole image.', 'A slope, a small step, and a step too far. Derivatives with actual numbers.', 'Every operation leaves a trail. The chain rule, PyTorch, and a first training loop.', 'Dozens to over a hundred Gaussians reconstruct an image. Initialization, detail, and saving the scene.'],
};
export const lessonNumbers = blocks.flatMap(block => block.lessons);
export const blockForLesson = number => blocks.find(block => block.lessons.includes(number));
export const pad = number => String(number).padStart(2, '0');
export function parseRoute(hash) {
  const match = hash.match(/^#\/(ru|en)(?:\/(\d+))?(?:\?.*)?$/);
  return {language: match?.[1], number: match && lessonNumbers.includes(Number(match[2])) ? Number(match[2]) : 0};
}
