"""Export the complete bilingual lesson narrative and saved figures for the web."""
import base64, json, pathlib, re, shutil

ROOT = pathlib.Path(__file__).resolve().parents[1]
COURSE = ROOT.parent
OUT = ROOT / 'dist'
REPOSITORY = 'https://github.com/HeinrichWirth/gaussian-splatting-course'
LABS = {1:(5,'scene'),2:(3,'bell'),3:(3,'plane'),4:(3,'covariance'),5:(3,'editor')}
LABS.update({6:(3,'loss'),7:(3,'descent'),8:(3,'autograd'),9:(3,'fitting')})

def rewrite_course_links(text, lang):
    """Keep lesson navigation local and preserve README sections on GitHub."""
    text = re.sub(r'\((?:\.\./en/|\.\./ru/)?lesson-(\d+)\.ipynb\)',
                  lambda match: f'(#/{lang}/{int(match[1])})', text)

    def readme_link(match):
        filename, anchor = match.groups()
        if anchor:
            return f'({REPOSITORY}/blob/main/{filename}{anchor})'
        return f'(#/{lang})'

    return re.sub(r'\(\.\./\.\./(README(?:_ru)?\.md)(#[^\s)]*)?\)',
                  readme_link, text)

def adapt(text, lang, lesson):
    text = re.sub(r'^\[(?:План курса|Программа курса|Course plan|Course outline|Course programme|Course program|Course overview)[^\n]*\n?', '', text, flags=re.M)
    if lesson == 1:
        text = text.replace('Кнопка «Сбросить»', 'Кнопка «Исходные параметры»')
        text = text.replace('The “Reset” button', 'The “Initial parameters” button')
    if lesson == 2:
        text = text.replace('Для начала хватит обычной арифметики и умения запустить ячейку Python.', 'Для начала хватит обычной арифметики.')
        text = text.replace('Ordinary arithmetic and the ability to run a Python cell are enough to get started.', 'Ordinary arithmetic is enough to get started.')
        text = text.replace('Проверим это в следующей ячейке.', 'Короткий пример на Python даёт те же числа.')
        text = text.replace('The next cell checks this numerically.', 'A short Python example checks this numerically.')
        text = re.sub(r'В каждом опыте меняется один параметр; остальные остаются исходными\.[^\n]*', 'В каждом опыте меняется один параметр; остальные остаются исходными. На графике в начале урока эти изменения видны сразу.', text)
        text = re.sub(r'Each experiment changes one parameter while keeping the others at their defaults\.[^\n]*', 'Each experiment changes one parameter while keeping the others at their defaults. The plot at the start of the lesson shows these changes directly.', text)
    if lesson == 3:
        replacements = {
            'ru': {
                'Начнём с трёх картинок одной функции.': 'Начнём с карты, холма и срезов одной функции.',
                '## 1. Одно пятно на трёх картинках': '## 1. Одно пятно: цвет, высота и срезы',
                'Слева вес показан цветом: тёмные участки слабые, светлые сильные. В середине тот же вес поднят вверх и получился холм. Справа проведены линии, на которых вес одинаков.': 'На карте вес показан цветом: тёмные участки слабые, светлые сильные. Линии на этой карте соединяют точки одинакового веса. Поле «Второй вид» переключает холм и два среза. На холме тот же вес показан высотой.',
                'Крестик отмечает точку, в которой смотрим вес. Небольшой плюс отмечает центр пятна.': 'Оранжевая точка отмечает место расчёта веса. Белый плюс отмечает центр пятна.',
                'Крестик пока стоит на прежнем месте, поэтому число под ним изменится.': 'Оранжевая точка пока стоит на прежнем месте, поэтому вес в ней изменится.',
                'У всех трёх видов одна область координат: от −5 до 5.': 'Диапазон координат во всех режимах — от −5 до 5. На срезах по горизонтали идёт координата, а по вертикали — вес.',
                'На тепловой карте этому числу соответствует цвет под крестиком.': 'На тепловой карте этому числу соответствует цвет у оранжевой точки.',
                'На контурном рисунке точка лежит между линиями': 'На той же карте точка лежит между линиями',
                'Справа на первом рисунке проведены линии одинакового веса.': 'На тепловой карте проведены линии одинакового веса.',
                '## 8. Разрежем пятно и узнаем старый колокол\n\n': '## 8. Разрежем пятно и узнаем старый колокол\n\nРежим «Два среза» в поле «Второй вид» показывает горизонтальное и вертикальное сечения через выбранную точку. Пунктирные кривые проходят через центр пятна.\n\n',
            },
            'en': {
                "We'll start with three pictures of the same function.": "We'll start with a map, a hill, and slices of the same function.",
                '## 1. One Spot in Three Pictures': '## 1. One Spot: Color, Height, and Slices',
                'On the left, weight is shown as color: dark areas are weak, bright areas are strong. In the middle, the same weight is drawn as height, making a hill. On the right, lines join points with equal weight.': 'The map shows weight as color: dark areas are weak, bright areas are strong. Contours on the map join points of equal weight. The “Second view” menu switches between a hill and two slices. The hill draws the same weight as height.',
                "The cross marks the point where we're reading the weight. A small plus marks the spot's center.": "The orange point marks the calculation location. A white plus marks the spot's center.",
                'The cross stays where it was, so the weight under it changes.': 'The orange point stays where it was, so the weight at that point changes.',
                'All three views use the same coordinate range, from −5 to 5.': 'Coordinates range from −5 to 5 in every view. A slice has a coordinate on its horizontal axis and weight on its vertical axis.',
                'On the heatmap, this number determines the color under the cross.': 'On the heatmap, this number determines the color at the orange point.',
                'In the contour view, the point lies between the lines': 'On the same map, the point lies between the lines',
                "The first figure's right-hand panel contains lines of equal weight.": 'Equal-weight contours are drawn on the heatmap.',
                '## 8. A Slice Through the Spot Reveals the Familiar Bell Curve\n\n': '## 8. A Slice Through the Spot Reveals the Familiar Bell Curve\n\nThe “Two slices” option under “Second view” shows horizontal and vertical sections through the selected point. The dashed reference curves pass through the spot’s center.\n\n',
            },
        }
        for before, after in replacements[lang].items():
            text = text.replace(before, after)
    if lesson == 4:
        text = text.replace('А рядом появятся четыре числа в квадратных скобках', 'Под рисунком появятся четыре числа в квадратных скобках')
        text = text.replace('Beside it, four numbers appear inside square brackets', 'Below the plot, four numbers appear inside square brackets')
        text = text.replace('Синяя линия соединяет точки с весом', 'Контур соединяет точки с весом')
        text = text.replace('The blue line joins points with weight', 'The contour joins points with weight')
        text = text.replace('Each entry in the table beside the ellipse now has a geometric meaning.', 'Each entry in the covariance matrix now has a geometric meaning.')
        text = text.replace('Следующая ячейка ловит ошибку через `try/except`: неправильные примеры не прерывают весь ноутбук.', 'В примере на Python `try/except` перехватывает ошибку: проверка продолжается и для остальных матриц.')
        text = text.replace('The next cell catches errors with `try/except`, so invalid examples do not stop the notebook.', 'The Python example catches errors with `try/except`, so the remaining matrices can still be checked.')
    if lesson == 5:
        text = re.sub(r'После выбора в поле \*\*«Пример»\*\* кнопка \*\*«Загрузить пример»\*\* открывает', 'Этот пример открывает', text)
        text = text.replace('«Выгрузить JSON»', '«Скачать JSON»').replace('“Export JSON”', '“Download JSON”')
        text = text.replace('«Загрузить JSON»', '«Применить JSON из поля»').replace('“Import JSON”', '“Apply JSON from field”')
        text = text.replace('Над ним появляется ссылка для скачивания `gaussian-scene.json`.', 'Тот же набор параметров скачивается файлом `gaussian-scene.json`.')
        text = text.replace('A link above it offers `gaussian-scene.json` as a download.', 'The same parameters can be downloaded as `gaussian-scene.json`.')
    if lesson == 7:
        replacements = {
            'Слева — яркость на прямой:': 'Сверху — яркость на прямой:',
            'Пунктир — образец с центром 1, сплошная линия — наше пятно.': 'Синяя кривая — образец с центром 1, оранжевая — наше пятно.',
            'Справа — ошибка:': 'Снизу — ошибка:',
            'Каждая точка правого графика означает **целый левый график**': 'Каждая точка нижнего графика означает **целый верхний график**',
            'Оранжевая точка справа — текущее состояние, пустой кружок — следующее после одного обновления.': 'Оранжевая точка на графике ошибки — текущее состояние. В таблице под графиками показан следующий шаг.',
            'The left plot shows intensity along a line:': 'The upper plot shows intensity along a line:',
            'The dashed curve is the target centered at 1; the solid curve is our spot.': 'The blue curve is the target centered at 1; the orange curve is our spot.',
            'The right plot shows loss:': 'The lower plot shows loss:',
            'Every point on the right represents **an entire left-hand curve**': 'Every point below represents **an entire curve above**',
            'The orange point on the right is the current state; the hollow circle is the next state after one update.': 'The orange point on the loss plot is the current state. The table beneath the plots shows the next update.',
        }
        for before, after in replacements.items():
            text = text.replace(before, after)
    if lesson == 8 and text.startswith('## 1.'):
        if lang == 'ru':
            text = '''## 1. Колокол начинает двигаться сам

В демонстрации два связанных опыта. Схема наверху прослеживает одну точку: от вычитания центра до квадрата ошибки. У каждого узла показаны результат операции и производная. Поля в группе «Одна точка · цепочка» меняют только этот расчёт.

График ниже обучает центр по всей кривой. Синяя линия — образец с центром $1.5$ и шириной $1$, оранжевая — наша попытка. В начале её центр равен $-1$. Высота и ширина закреплены; меняется только центр.

В группе «Вся кривая · обучение» задаются начальный центр и размер шага. «Один шаг обучения» выполняет одно обновление, «Обучение» продолжает последовательность, «Сброс обучения» возвращает выбранное начало. Рядом с графиком видны номер шага, текущий центр и ошибка. При нулевом размере шага ничего не движется; при слишком большом возможен перелёт. Числовой разбор дальше начнётся с одной точки, а затем соединит все 41 измерение.'''
        else:
            text = '''## 1. The Bell Curve Starts Moving on Its Own

The demonstration has two related experiments. The diagram at the top follows one sample, from subtracting the center to squaring the residual. Each node shows its forward value and derivative. Controls in “One point · chain” change only that calculation.

The plot below fits the center using the whole curve. Blue is the target with center $1.5$ and scale $1$; orange is our attempt, starting at $-1$. Height and scale stay fixed; only the center changes.

“Whole curve · training” sets the initial center and learning rate. “One training step” makes one update, “Train” continues the sequence, and “Reset training” returns to the chosen start. The step count, current center, and loss appear beside the plot. A zero rate leaves the curve still; an excessive step can overshoot. The numerical explanation starts with one sample, then connects all 41 measurements.'''
    if lesson == 9:
        text = text.replace('Поле «Образец» переключает фигуры, цветок и мелкие клетки.', 'Поле «Картинка» переключает образцы; рядом доступна загрузка своей картинки.')
        text = text.replace('Смена параметров задаёт новый опыт; «Обучение» строит для него новую последовательность. Сцена в JSON внизу содержит уже обученные пятна.', 'Смена образца или начальных настроек начинает новый опыт. «Обучение» продолжает обновления, «Пауза» останавливает их, «Заново» возвращает начало. Размер шага можно менять по ходу. «Скачать сцену JSON» сохраняет текущие пятна для редактора урока 5.')
        text = text.replace('“Target” switches between shapes, a flower, and a fine checker pattern.', '“Image” switches between targets; a local image can also be loaded.')
        text = text.replace('Changing settings defines a new experiment; “Train” builds a new sequence for it. The JSON scene at the bottom contains the trained spots.', 'Changing the target or initialization starts a new experiment. “Train” continues the updates, “Pause” stops them, and “Restart” returns to the beginning. The learning rate can change during a run. “Download scene JSON” saves the current spots for the lesson 5 editor.')
    return rewrite_course_links(text, lang)


def main():
    for lang in ['ru','en']:
        lessons=[]
        for number in range(1,10):
            path=COURSE/'notebooks'/lang/f'lesson-{number:02}.ipynb'
            notebook=json.loads(path.read_text(encoding='utf8'))
            tagged = any(any(tag.startswith('web-lab:') for tag in cell.get('metadata', {}).get('tags', []))
                         for cell in notebook['cells'])
            cells=[]
            title=''
            for i,cell in enumerate(notebook['cells']):
                src=''.join(cell['source'])
                if cell['cell_type']=='markdown':
                    if i==0:
                        title=src.splitlines()[0].lstrip('# ').strip()
                        src='\n'.join(src.splitlines()[1:])
                        src=re.sub(r'^\s*\[[^\n]+\n', '', src)
                    cells.append({'type':'text','source':adapt(src,lang,number),'cell':i})
                    continue
                if '%matplotlib inline' in src: continue
                lab_tag = next((tag.split(':', 1)[1] for tag in cell.get('metadata', {}).get('tags', []) if tag.startswith('web-lab:')), None)
                if lab_tag or (not tagged and i==LABS[number][0]):
                    if number == 5:
                        for j, output in enumerate(cell.get('outputs', [])):
                            if 'image/png' in output.get('data', {}):
                                name=f'assets/{lang}/lesson-{number:02}-{i:02}-{j}.png'
                                dest=OUT/name; dest.parent.mkdir(parents=True,exist_ok=True)
                                dest.write_bytes(base64.b64decode(''.join(output['data']['image/png'])))
                                cells.append({'type':'image','src':name,'cell':i})
                    cells.append({'type':'lab','name':lab_tag or LABS[number][1],'cell':i})
                else:
                    for j,output in enumerate(cell.get('outputs',[])):
                        data=output.get('data',{})
                        if 'image/png' in data:
                            name=f'assets/{lang}/lesson-{number:02}-{i:02}-{j}.png'
                            dest=OUT/name; dest.parent.mkdir(parents=True,exist_ok=True)
                            dest.write_bytes(base64.b64decode(''.join(data['image/png'])))
                            cells.append({'type':'image','src':name,'cell':i})
                outputs=[]
                for output in cell.get('outputs',[]):
                    if output.get('output_type')=='stream': outputs.append(''.join(output['text']))
                    elif 'text/plain' in output.get('data',{}) and not any(k in output['data'] for k in ['image/png','text/html']):
                        value=''.join(output['data']['text/plain'])
                        if not value.startswith(('<','VBox','HBox')): outputs.append(value)
                cells.append({'type':'code','source':src,'output':'\n'.join(outputs),'cell':i})
            lessons.append({'number':number,'block':1 if number <= 5 else 2,'title':title,'cells':cells,'source':f'https://github.com/HeinrichWirth/gaussian-splatting-course/blob/main/notebooks/{lang}/lesson-{number:02}.ipynb'})
        (OUT/'content').mkdir(exist_ok=True)
        (OUT/'content'/f'{lang}.json').write_text(json.dumps(lessons,ensure_ascii=False),encoding='utf8')
    print('Exported all nine lessons in Russian and English.')


if __name__ == "__main__":
    main()
