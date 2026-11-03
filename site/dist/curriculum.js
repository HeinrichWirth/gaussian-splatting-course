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
  {number: 5, lessons: [22, 23, 24, 25, 26],
    title: {ru: 'Когда сцена движется', en: 'When the scene moves'},
    short: {ru: 'Движение', en: 'Motion'},
    description: {ru: 'На двух кадрах пятно оказалось в разных местах. Двигалась камера или само пятно? От двух наблюдений дойдём до сцены, в которой можно менять время и ракурс по отдельности.', en: 'A splat appears in different places in two frames. Did the camera move, or did the splat? Two observations lead to a scene with separate controls for time and viewpoint.'}},
  {number: 6, lessons: [27, 28, 29, 30],
    title: {ru: 'Сколько весит сцена', en: 'How much space a scene takes'},
    short: {ru: 'Размер и скорость', en: 'Size and speed'},
    description: {ru: 'Сохраним одну сцену несколькими способами. Для каждого варианта посчитаем байты, измерим время и посмотрим, какие детали исчезли из кадра.', en: 'The same scene can be stored several ways. For each version we count the bytes, measure the time, and look at which details disappear from the image.'}},
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
  ['Кто движется в кадре?', 'What moved in the frame?',
    'Сдвиг камеры и сдвиг пятна дают похожий кадр. Что добавляет вторая камера?', 'Camera motion and object motion can look alike. What can a second camera tell us?'],
  ['У пятна появилась траектория', 'A splat gets a trajectory',
    'Центр, поворот и ширина меняются со временем. Между измерениями и за их пределами прогноз ведёт себя по-разному.', 'Center, rotation, and width change over time. A prediction between observations behaves differently beyond the last one.'],
  ['Соседи между кадрами', 'Neighbors across frames',
    'Одни и те же пятна поворачиваются вместе. Связь с соседом помогает восстановить пропущенное движение.', 'The same splats turn together. A visible neighbor helps recover motion hidden between frames.'],
  ['Как гнётся исходная сцена', 'How a canonical scene bends',
    'Сетка хранит несколько значений. Между её узлами получается непрерывное смещение.', 'A grid stores a handful of values. Interpolation turns them into a continuous displacement.'],
  ['Движущаяся сцена своими силами', 'Fitting a moving scene',
    'Два ракурса дают наблюдения для обучения. Другой ракурс и промежуточный момент остаются для проверки.', 'Two views supply the observations. Another camera and an intermediate time put the fit to the test.'],
  ['От пятна до байта', 'From a splat to a byte',
    'Центр, форма, цвет и непрозрачность занимают место. Файл и рабочая память дают разные числа.', 'Center, shape, color, and opacity all take space. File size and working memory give different totals.'],
  ['Что можно убрать из скульптуры', 'What the sculpture can lose',
    'Тот же ракурс после удаления пятен и округления чисел. Сколько байтов стоит потерянная деталь?', 'The same view after pruning and rounding. How many bytes did the missing detail save?'],
  ['Как записать движение', 'Storing motion',
    'Один и тот же путь хранится целыми кадрами или изменениями между ними. Ошибку видно на траектории.', 'A path can be stored as complete frames or changes between them. The trajectory shows where the errors accumulate.'],
  ['Выбор под ограничение', 'Choosing within a budget',
    'У сцены есть лимит в байтах. Варианты проходят один опыт, а выбранный проверяется на отложенном снимке.', 'A scene has a byte budget. The candidates go through the same trial, followed by a held-out view of the chosen model.'],
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
