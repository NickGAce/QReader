import cv2
from qreader import QReader
from pyzbar.pyzbar import decode, ZBarSymbol
import glob
import os
import numpy as np
import shutil
import time
import logging
from datetime import datetime
import argparse


def setup_logging(verbose=False, log_folder="logs", log_to_file=True):
    """Настройка системы логирования"""
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)

    log_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    # Логирование в консоль всегда
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_formatter)
    logger.addHandler(console_handler)

    # Логирование в файл только если log_to_file=True
    if log_to_file:
        # Создаем папку для логов, если ее нет
        os.makedirs(log_folder, exist_ok=True)

        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(log_folder, f"qr_processor_{current_time}.log")
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(log_formatter)
        logger.addHandler(file_handler)

        logger.info(f"Логи будут сохранены в: {log_file}")
    else:
        logger.info("Логи будут выводиться только в консоль")

    return logger

def preprocess_image(image_path, use_clahe=False, logger=None):
    """Предварительная обработка изображения"""
    if logger:
        logger.debug(f"Начало предварительной обработки изображения: {image_path}")

    try:
        image = cv2.imread(image_path)
        if image is None:
            if logger:
                logger.error(f"Не удалось загрузить изображение: {image_path}")
            return None

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        if use_clahe:
            if logger:
                logger.debug("Применение CLAHE для улучшения контраста")
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            gray = clahe.apply(gray)

        if logger:
            logger.debug(f"Успешная предварительная обработка изображения: {image_path}")

        return gray

    except Exception as e:
        if logger:
            logger.error(f"Ошибка при предварительной обработке изображения {image_path}: {str(e)}")
        return None

def read_qr_qreader(image, logger=None):
    """Читает QR-код с помощью QReader"""
    if logger:
        logger.debug("Начало распознавания QR-кода с помощью QReader")

    try:
        qreader = QReader()

        # Попытка на оригинальном изображении
        result = qreader.detect_and_decode(image)
        if result not in [(), (None, None), (None,)]:
            if logger:
                logger.debug(f"QReader успешно распознал QR-код: {result}")
            return result

        if logger:
            logger.debug("QReader не распознал QR-код на оригинальном изображении, пробуем масштабирование")

        # Пробуем увеличенное изображение
        for scale in [1.5, 2.0, 3.0]:
            scaled_image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
            result = qreader.detect_and_decode(scaled_image)
            if result not in [(), (None, None), (None,)]:
                if logger:
                    logger.debug(f"QReader успешно распознал QR-код с масштабированием {scale}x: {result}")
                return result

        if logger:
            logger.debug("QReader не распознал QR-код при масштабировании, пробуем повороты")

        # Пробуем повороты
        for angle in [90, 180, 270]:
            rotated = cv2.rotate(image, {90: cv2.ROTATE_90_CLOCKWISE,
                                         180: cv2.ROTATE_180,
                                         270: cv2.ROTATE_90_COUNTERCLOCKWISE}[angle])
            result = qreader.detect_and_decode(rotated)
            if result not in [(), (None, None), (None,)]:
                if logger:
                    logger.debug(f"QReader успешно распознал QR-код с поворотом {angle}°: {result}")
                return result

        if logger:
            logger.debug("QReader не смог распознать QR-код после всех попыток")
        return None

    except Exception as e:
        if logger:
            logger.error(f"Ошибка в QReader: {str(e)}")
        return None

def read_qr_pyzbar(image, logger=None):
    """Читает QR-код с помощью pyzbar"""
    if logger:
        logger.debug("Начало распознавания QR-кода с помощью PyZbar")

    try:
        decoded_objects = decode(image, symbols=[ZBarSymbol.QRCODE])

        if not decoded_objects:
            if logger:
                logger.debug("PyZbar не распознал QR-код на оригинальном изображении, пробуем масштабирование")

            # Пробуем увеличение
            for scale in [1.5, 2.0, 3.0]:
                scaled_image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
                decoded_objects = decode(scaled_image, symbols=[ZBarSymbol.QRCODE])
                if decoded_objects:
                    if logger:
                        logger.debug(f"PyZbar успешно распознал QR-код с масштабированием {scale}x")
                    break

        if not decoded_objects:
            if logger:
                logger.debug("PyZbar не распознал QR-код при масштабировании, пробуем повороты")

            # Пробуем повороты
            for angle in [90, 180, 270]:
                rotated = cv2.rotate(image, {90: cv2.ROTATE_90_CLOCKWISE,
                                             180: cv2.ROTATE_180,
                                             270: cv2.ROTATE_90_COUNTERCLOCKWISE}[angle])
                decoded_objects = decode(rotated, symbols=[ZBarSymbol.QRCODE])
                if decoded_objects:
                    if logger:
                        logger.debug(f"PyZbar успешно распознал QR-код с поворотом {angle}°")
                    break

        result = [obj.data.decode("utf-8") for obj in decoded_objects] if decoded_objects else None

        if logger:
            if result:
                logger.debug(f"PyZbar успешно распознал QR-код: {result}")
            else:
                logger.debug("PyZbar не смог распознать QR-код после всех попыток")

        return result

    except Exception as e:
        if logger:
            logger.error(f"Ошибка в PyZbar: {str(e)}")
        return None

def detect_qr_sift(image, template_folder, logger=None):

    if logger:
        logger.debug("Начало поиска QR-кода с помощью SIFT")
        logger.debug(f"Размер входного изображения: {image.shape}")

    try:
        # Количество ключевых точек
        sift = cv2.SIFT_create(nfeatures=5000, contrastThreshold=0.03, edgeThreshold=10)
        kp1, des1 = sift.detectAndCompute(image, None)

        if des1 is None or len(kp1) < 10:
            if logger:
                logger.debug("Недостаточно ключевых точек в исходном изображении")
            return None

        if logger:
            logger.debug(f"Найдено ключевых точек: {len(kp1)}")

        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
        best_match = None
        best_quality = float('inf')

        for template_path in glob.glob(os.path.join(template_folder, "*.png")):
            if logger:
                logger.debug(f"\nПроверка эталонного изображения: {template_path}")

            template = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
            if template is None:
                if logger:
                    logger.warning(f"Не удалось загрузить эталон: {template_path}")
                continue

            for scale in [0.8, 1.0, 1.2]:
                # Масштабирование
                scaled = cv2.resize(template, None, fx=scale, fy=scale)

                for angle in [0, 90, 180, 270]:
                    # Реализация поворота
                    if angle == 0:
                        rotated = scaled.copy()
                    elif angle == 90:
                        rotated = cv2.rotate(scaled, cv2.ROTATE_90_CLOCKWISE)
                    elif angle == 180:
                        rotated = cv2.rotate(scaled, cv2.ROTATE_180)
                    elif angle == 270:
                        rotated = cv2.rotate(scaled, cv2.ROTATE_90_COUNTERCLOCKWISE)
                    else:
                        continue

                    kp2, des2 = sift.detectAndCompute(rotated, None)
                    if des2 is None or len(kp2) < 10:
                        if logger:
                            logger.debug(f"Не найдено ключевых точек в эталоне (масштаб {scale}, угол {angle})")
                        continue

                    matches = bf.knnMatch(des1, des2, k=2)
                    good = []
                    try:
                        for m, n in matches:
                            if m.distance < 0.7 * n.distance:
                                good.append(m)
                    except ValueError:
                        continue

                    if logger:
                        logger.debug(f"Масштаб: {scale}x, угол: {angle}° - совпадений: {len(good)}")

                    if len(good) > 15:
                        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
                        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

                        matrix, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 3.0)
                        if matrix is not None and mask.sum() > 15:
                            quality = sum(m.distance for m in good) / len(good)

                            if best_match is None or quality < best_quality:
                                best_quality = quality
                                best_match = {
                                    'path': template_path,
                                    'quality': quality,
                                    'matches': len(good),
                                    'inliers': mask.sum(),
                                    'scale': scale,
                                    'angle': angle
                                }

        if best_match and logger:
            logger.info(f"\nЛучшее совпадение: {best_match['path']}")
            logger.info(f"Качество: {best_match['quality']:.2f}")
            logger.info(f"Совпадений: {best_match['matches']}")
            logger.info(f"Inliers: {best_match['inliers']}")

        if best_match and best_match['inliers'] > 20:
            template_img = cv2.imread(best_match['path'], cv2.IMREAD_GRAYSCALE)
            qr_value = read_qr_qreader(template_img, logger) or read_qr_pyzbar(template_img, logger)

            if qr_value:
                return qr_value[0] if isinstance(qr_value, (list, tuple)) else qr_value

        if logger:
            logger.debug("Не найдено достаточно хороших совпадений")
        return None

    except Exception as e:
        if logger:
            logger.error(f"Ошибка в SIFT: {str(e)}", exc_info=True)
        return None

def get_unique_filename(folder, base_name, extension, logger=None):
    """Генерирует уникальное имя файла"""
    if logger:
        logger.debug(f"Генерация уникального имени для: {base_name}{extension} в папке {folder}")

    counter = 1
    while True:
        new_name = f"{base_name}_{counter}{extension}" if counter > 1 else f"{base_name}{extension}"
        if not os.path.exists(os.path.join(folder, new_name)):
            if logger:
                logger.debug(f"Сгенерировано уникальное имя: {new_name}")
            return new_name
        counter += 1

def parse_arguments():
    """Парсинг аргументов командной строки"""
    parser = argparse.ArgumentParser(        description='Скрипт для обработки QR-кодов в изображениях. '
                                            'Работает в несколько этапов для распознавания: QReader, PyZbar и SIFT.',
        epilog='Пример использования:\n'
               '  python ./name_script --input my_photos --output renamed\n'
               '  python ./name_script --input my_photos --output renamed --templates my_templates --failed bad_photos --logs my_logs --verbose',
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument('--input', default='photos', help='Папка с исходными изображениями (по умолчанию: photos)')
    parser.add_argument('--output', default='renamed_photos',
                        help='Папка для переименованных изображений (по умолчанию: renamed_photos)')
    parser.add_argument('--templates', default='qr_reference',
                        help='Папка с эталонными QR-кодами (по умолчанию: qr_reference)')
    parser.add_argument('--failed', default='failed_photos',
                        help='Папка для неудачных изображений (по умолчанию: failed_photos)')
    parser.add_argument('--logs', default='logs', help='Папка для логов (по умолчанию: logs)')
    parser.add_argument('--verbose', action='store_true', help='Включить подробное логирование')
    parser.add_argument('--log_to_file', action='store_true', help='Записывать логи в файл (по умолчанию: False)')

    return parser.parse_args()

def main():
    """Основная функция"""
    print("=== QR Code Processor ===")

    # Парсинг аргументов командной строки
    args = parse_arguments()

    # Установка путей из аргументов
    image_folder = args.input
    renamed_folder = args.output
    template_folder = args.templates
    failed_folder = args.failed
    log_folder = args.logs
    verbose = args.verbose
    log_to_file = args.log_to_file

    logger = setup_logging(verbose, log_folder, log_to_file)

    logger.info("Запуск QR Code Processor")
    logger.info(f"Режим подробного логирования: {'включен' if verbose else 'выключен'}")

    # Создаем папки, если их нет
    for folder in [renamed_folder, failed_folder]:
        os.makedirs(folder, exist_ok=True)
        logger.info(f"Создана (или уже существует) папка: {folder}")

    # Статистика
    qreader_result_count = 0
    pyzbar_result_count = 0
    sift_result_count = 0
    total = 0
    problem = []

    # Поиск файлов
    jpg_files = glob.glob(os.path.join(image_folder, "*.[jJ][pP][gG]"))
    jpg_files_len = len(jpg_files)
    logger.info(f"Найдено {jpg_files_len} JPEG файлов для обработки")

    for image_path in jpg_files:
        start_time = time.time()
        logger.info(f"\nНачало обработки изображения: {image_path}")

        # Загрузка и предобработка изображения
        logger.debug("Загрузка и предварительная обработка изображения")
        preprocessed_image = preprocess_image(image_path, True, logger)

        if preprocessed_image is None:
            logger.error(f"Не удалось обработать изображение: {image_path}")
            problem.append(image_path)
            continue

        # 1. Пробуем QReader
        logger.debug("Попытка распознавания с помощью QReader")
        qreader_start_time = time.time()
        qreader_result = read_qr_qreader(preprocessed_image, logger)
        qreader_time = time.time() - qreader_start_time
        logger.info(f"QReader результат: {qreader_result}, время: {qreader_time:.2f} сек")

        # 2. Если QReader не смог, пробуем PyZbar
        pyzbar_result = None
        if not qreader_result:
            logger.debug("Попытка распознавания с помощью PyZbar")
            pyzbar_start_time = time.time()
            pyzbar_result = read_qr_pyzbar(preprocessed_image, logger)
            pyzbar_time = time.time() - pyzbar_start_time
            logger.info(f"PyZbar результат: {pyzbar_result}, время: {pyzbar_time:.2f} сек")

        # 3. Если PyZbar не смог, пробуем SIFT
        sift_result = None
        if not qreader_result and not pyzbar_result:
            logger.debug("Попытка распознавания с помощью SIFT")
            sift_start_time = time.time()
            sift_result = detect_qr_sift(preprocessed_image, template_folder, logger)
            sift_time = time.time() - sift_start_time
            logger.info(f"SIFT результат: {sift_result}, время: {sift_time:.2f} сек")

        # Определяем значение QR-кода
        qr_value = None
        if qreader_result:
            qr_value = qreader_result[0] if isinstance(qreader_result, (list, tuple)) else qreader_result
            qreader_result_count += 1
        elif pyzbar_result:
            qr_value = pyzbar_result[0] if isinstance(pyzbar_result, (list, tuple)) else pyzbar_result
            pyzbar_result_count += 1
        elif sift_result:
            qr_value = sift_result
            sift_result_count += 1

        # Обработка результата
        if qr_value:
            total += 1
            base_name = qr_value
            if sift_result:
                base_name += "_check"

            new_image_name = get_unique_filename(renamed_folder, base_name, ".jpg", logger)
            new_image_path = os.path.join(renamed_folder, new_image_name)

            try:
                shutil.move(image_path, new_image_path)
                logger.info(f"Фото перемещено и переименовано: {new_image_path}")
            except Exception as e:
                logger.error(f"Ошибка при перемещении файла: {str(e)}")
                problem.append(image_path)
        else:
            failed_image_name = f"failed_{os.path.basename(image_path)}"
            failed_image_path = os.path.join(failed_folder, failed_image_name)

            try:
                shutil.move(image_path, failed_image_path)
                logger.info(f"Фото не распознано и перемещено в папку неудачных: {failed_image_path}")
                problem.append(image_path)
            except Exception as e:
                logger.error(f"Ошибка при перемещении неудачного файла: {str(e)}")
                problem.append(image_path)

        logger.info(f"Общее время обработки: {time.time() - start_time:.2f} сек")

    # Итоговые результаты
    logger.info("\n=== Итоговые результаты обработки ===")
    logger.info(f"Количество обработанных файлов: {jpg_files_len}")
    logger.info(f"Успешно распознано QReader: {qreader_result_count}")
    logger.info(f"Успешно распознано PyZbar: {pyzbar_result_count}")
    logger.info(f"Успешно распознано SIFT: {sift_result_count}")
    logger.info(f"Всего распознано: {total}")
    logger.info(f"Не распознано: {len(problem)}")
    time.sleep(1)

    if problem:
        logger.info("\nСписок проблемных файлов:")
        for file in problem:
            logger.info(file)
        time.sleep(1)


    print("\nОбработка завершена.")

if __name__ == "__main__":
    main()