
def shelf_report(
    detections: list[dict],
    frame_width: int,
    frame_height: int,
    conf_threshold: float,
    alert_share: float,
) -> dict:
    """Отчёт по кадру: сколько пустых мест на полке и какую долю кадра они занимают.

    Возвращает словарь с ключами count, share и alert.
    """
    new_detections = {"count": 0, "share": 0.0, "alert": False}
    frame_area = frame_height * frame_width

    for detection in detections:
        if detection["conf"] < conf_threshold:
            continue
        if detection["label"] != "empty_space":
            continue
        new_detections["count"] += 1
        xmin, ymin, xmax, ymax = detection["box"]

        # Вычисление ширины и высоты
        width = xmax - xmin
        height = ymax - ymin

        # Расчет площади
        area = width * height
        new_detections["share"] += area / frame_area

    if new_detections["share"] > alert_share:
        new_detections["alert"] = True
    new_detections["share"] = round(new_detections["share"], 4)
    print(new_detections)
    return new_detections



if __name__ == "__main__":
    detections = [
        {"label": "empty_space", "conf": 0.81, "box": [96, 224, 128, 272]},
        {"label": "empty_space", "conf": 0.42, "box": [400, 208, 440, 256]},
        {"label": "product", "conf": 0.95, "box": [0, 0, 64, 64]},
    ]

    shelf_report(detections, 640, 640, 0.4, 0.005)