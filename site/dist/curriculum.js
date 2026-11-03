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
  {number: 3, lessons: [10, 11, 12, 13, 14, 15],
    title: {ru: 'Как трёхмерная сцена становится картинкой', en: 'How a 3D scene becomes an image'},
    short: {ru: 'Сцена и камера', en: 'Scene and camera'},
    description: {ru: 'Обойдём один эллипсоид с камерой. Как выбрать пиксель для его центра, какой формы получится пятно и что произойдёт, когда одно пятно закроет другое?', en: 'A camera moves around an ellipsoid. Where does its center land, what shape does the footprint take, and what happens when two footprints overlap?'}},
  {number: 4, lessons: [16, 17, 18, 19, 20, 21],
    title: {ru: 'Статическая сцена по фотографиям', en: 'A static scene from photographs'},
    short: {ru: 'Реконструкция', en: 'Reconstruction'},
    description: {ru: 'Сначала известны и камеры, и сама сцена. Затем останутся только фотографии. Один снимок отложим, чтобы увидеть, что получилось с нового ракурса.', en: 'At first both the cameras and the scene are known. Then we work from photographs, keeping one view aside to see what the reconstruction looks like from there.'}},
];
export const names = {
  ru: ['Из пятен — в картинку', 'Один колокол', 'Гауссиана на плоскости', 'Поворот и матрица формы', 'Несколько пятен', 'Как измерить ошибку', 'Куда двигать параметр', 'От цепочки к обучению', 'Восстанавливаем картинку'],
  en: ['From splats to an image', 'One bell curve', 'A Gaussian on a plane', 'Rotation and the shape matrix', 'Several splats', 'Measuring the error', 'Which way to move', 'From a chain to learning', 'Reconstructing an image'],
};
export const descriptions = {
  ru: ['Сцена, камера и 192 гауссианы. Что вообще хранится внутри.', 'Центр, ширина и вес. Одна формула — пять простых действий.', 'Два направления, цветная карта и холм. От круга к эллипсу.', 'Свои оси пятна, поворот и четыре числа ковариации.', 'Цвет, прозрачность и порядок. Маленький редактор сцены.', 'Образец, результат и карта различий. От четырёх чисел до ошибки всего изображения.', 'Наклон графика, маленький шаг и слишком большой шаг. Производная на конкретных числах.', 'Каждая операция оставляет след. Правило цепочки, PyTorch и первый цикл обучения.', 'Десятки и сотня гауссиан восстанавливают изображение. Начальные параметры, детали и сохранение сцены.'],
  en: ['A scene, a camera, and 192 Gaussians. What is actually stored.', 'Center, width, and weight. One formula in five simple steps.', 'Two directions, a heatmap, and a hill. From a circle to an ellipse.', 'Local axes, rotation, and the four entries of covariance.', 'Color, opacity, and order. A small scene editor.', 'Target, prediction, and a difference map. From four numbers to a loss for the whole image.', 'A slope, a small step, and a step too far. Derivatives with actual numbers.', 'Every operation leaves a trail. The chain rule, PyTorch, and a first training loop.', 'Dozens to over a hundred Gaussians reconstruct an image. Initialization, detail, and saving the scene.'],
};

const sceneLessons = [
  ['Точка, мир и камера', 'Point, world, and camera',
    'Одна точка в двух системах координат. Сдвиг и поворот по отдельности.', 'One point described in two frames. Translation and rotation, one at a time.'],
  ['От луча до пикселя', 'From a ray to a pixel',
    'Две точки на разной глубине. Откуда в перспективе берётся деление на z.', 'Two points at different depths reveal where division by z comes from.'],
  ['Объёмное пятно', 'A Gaussian in space',
    'Три ширины задают форму. Ось, угол и четыре числа задают поворот.', 'Three widths set the shape. An axis, an angle, and four numbers describe its rotation.'],
  ['След эллипсоида на экране', 'An ellipsoid on the screen',
    'Якобиан переводит небольшое смещение в пиксели. Что происходит вблизи камеры?', 'The Jacobian translates a small displacement into pixels. What happens up close?'],
  ['Как складывается кадр', 'How a frame comes together',
    'Два пятна закрывают друг друга. Сколько света от каждого дойдёт до пикселя?', 'Two splats overlap. How much of each color reaches the pixel?'],
  ['Цвет зависит от ракурса', 'Color changes with the view',
    'Камера обходит пятно. Направление взгляда меняет вклад цветовых коэффициентов.', 'Moving around a splat changes how its color coefficients contribute to the image.'],
  ['Одна сцена, несколько снимков', 'One scene, several views',
    'Из одной камеры виден цвет. Вторая помогает найти глубину.', 'One camera sees the color. A second view helps recover depth.'],
  ['Когда одного пятна мало', 'When one splat is not enough',
    'Клонирование, деление и удаление меняют рисунок и число параметров.', 'Cloning, splitting, and pruning change both the image and the number of parameters.'],
  ['Как ускоряется рендеринг', 'How rendering gets faster',
    'Один кадр двумя способами. Сколько проверок экономит разбиение экрана на плитки?', 'One frame drawn two ways. How many checks disappear when the screen is divided into tiles?'],
  ['От снимков до камер', 'From photographs to cameras',
    'Одна деталь на двух снимках, два луча и точка в пространстве. Так начинается COLMAP.', 'A detail in two photographs, two rays, and a point in space. The starting point for COLMAP.'],
  ['Сцена по фотографиям', 'A scene from photographs',
    'Снимки, начальные точки, обучение и сохранённая сцена для нового ракурса.', 'Photographs, initial points, a training run, and a saved scene to view from a new angle.'],
  ['Что выдаёт ошибку', 'Where a reconstruction goes wrong',
    'Отложенный снимок, карта ошибок и цена кадра. Где расходятся модель и фотография?', 'A held-out photograph, an error map, and the cost of a frame. Where does the model disagree with the photo?'],
];

for (const [ru, en, ruDescription, enDescription] of sceneLessons) {
  names.ru.push(ru);
  names.en.push(en);
  descriptions.ru.push(ruDescription);
  descriptions.en.push(enDescription);
}
export const lessonNumbers = blocks.flatMap(block => block.lessons);
export const blockForLesson = number => blocks.find(block => block.lessons.includes(number));
export const pad = number => String(number).padStart(2, '0');
export function parseRoute(hash) {
  const match = hash.match(/^#\/(ru|en)(?:\/(\d+))?(?:\?.*)?$/);
  return {language: match?.[1], number: match && lessonNumbers.includes(Number(match[2])) ? Number(match[2]) : 0};
}
