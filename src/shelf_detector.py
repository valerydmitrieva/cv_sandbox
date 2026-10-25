import math


def plan_fpn_levels(
    boxes: list[list[float]],
    frame_width: int,
    frame_height: int,
    input_size: int,
) -> dict:
    """Раскладка размеченных объектов по уровням пирамиды признаков P2-P5.

    Координаты рамок заданы в пикселях исходного кадра. Кадр растягивается
    до квадрата input_size × input_size без сохранения пропорций.
    Тонкой считается рамка с меньшей стороной строго меньше 32 пикселей.
    Возвращает словарь с ключами levels, tiny, unused и min_input.
    Для пустой разметки min_input равен 320.
    """
    levels: dict[int, int] = {2: 0, 3: 0, 4: 0, 5: 0}
    relative_sides: list[tuple[float, float]] = []
    tiny = 0


    for box in boxes:
        xmin, ymin, xmax, ymax = box
        relative_sides.append(
            ((xmax - xmin) / frame_width, (ymax - ymin) / frame_height)
        )

    for relative_width, relative_height in relative_sides:
        width = relative_width * input_size
        height = relative_height * input_size
        level = min(
            5,
            max(2, math.floor(4 + math.log2(math.sqrt(width * height) / 224))),
        )
        levels[level] += 1
        if min(width, height) < 32:
            tiny += 1

    min_input = None
    for candidate in (320, 512, 640, 800, 1024, 1280):
        thin_count = sum(
            min(width * candidate, height * candidate) < 32
            for width, height in relative_sides
        )
        if thin_count * 4 <= len(relative_sides):
            min_input = candidate
            break

    return {
        "levels": levels,
        "tiny": tiny,
        "unused": sorted(level for level, count in levels.items() if count == 0),
        "min_input": min_input,
    }

if __name__ == "__main__":
    boxes = [
        [96, 224, 130, 279],
        [400, 208, 417, 254],
        [40, 40, 340, 340],
    ]

    assert plan_fpn_levels(boxes, 640, 640, 640) == {
        "levels": {2: 2, 3: 0, 4: 1, 5: 0},
        "tiny": 1,
        "unused": [3, 5],
        "min_input": 1280,
    }

    assert plan_fpn_levels(boxes, 640, 640, 320) == {
        "levels": {2: 2, 3: 1, 4: 0, 5: 0},
        "tiny": 2,
        "unused": [4, 5],
        "min_input": 1280,
    }

