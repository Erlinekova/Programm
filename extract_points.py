#!/usr/bin/env python3
"""
extract_points.py - Извлечение 3D точек из формата COLMAP

Парсит файлы points3D.txt или points3D.bin из проекта COLMAP
и экспортирует координаты точек в массив NumPy.

Использование:
    python extract_points.py <colmap_path> [output_file] [--format txt|bin]

Примеры:
    python extract_points.py ./colmap_project
    python extract_points.py ./colmap_project points.npy --format bin
    python extract_points.py ./colmap_project points.npz --with-colors
"""

import argparse
import struct
import sys
from pathlib import Path
from typing import Optional, Tuple

import numpy as np


def parse_colmap_txt(points_file: Path) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Парсинг текстового формата COLMAP (points3D.txt)
    
    Формат файла:
    # 3D point list with one line of data per point:
    #   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)
    """
    points_3d = []
    colors = []
    
    with open(points_file, 'r') as f:
        for line in f:
            # Пропускаем комментарии и пустые строки
            if line.startswith('#') or not line.strip():
                continue
            
            parts = line.split()
            if len(parts) < 8:
                continue
            
            # Извлекаем координаты X, Y, Z
            x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
            points_3d.append([x, y, z])
            
            # Извлекаем цвета R, G, B
            r, g, b = int(parts[4]), int(parts[5]), int(parts[6])
            colors.append([r, g, b])
    
    points_array = np.array(points_3d, dtype=np.float64)
    colors_array = np.array(colors, dtype=np.uint8) if colors else None
    
    return points_array, colors_array


def parse_colmap_bin(points_file: Path) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Парсинг бинарного формата COLMAP (points3D.bin)
    
    Структура записи:
    - POINT3D_ID: uint64
    - X, Y, Z: double (float64)
    - R, G, B: uint8
    - ERROR: double (float64)
    - TRACK_LENGTH: uint64
    - TRACK: (IMAGE_ID uint32, POINT2D_IDX uint32) * TRACK_LENGTH
    """
    points_3d = []
    colors = []
    
    with open(points_file, 'rb') as f:
        # Читаем количество точек
        num_points = struct.unpack('<Q', f.read(8))[0]
        
        for _ in range(num_points):
            # POINT3D_ID
            point_id = struct.unpack('<Q', f.read(8))[0]
            
            # X, Y, Z
            x, y, z = struct.unpack('<ddd', f.read(24))
            points_3d.append([x, y, z])
            
            # R, G, B
            r, g, b = struct.unpack('<BBB', f.read(3))
            colors.append([r, g, b])
            
            # ERROR
            error = struct.unpack('<d', f.read(8))[0]
            
            # TRACK_LENGTH
            track_length = struct.unpack('<Q', f.read(8))[0]
            
            # Пропускаем TRACK (IMAGE_ID, POINT2D_IDX) * track_length
            f.read(track_length * 8)  # 2 * uint32 * track_length
    
    points_array = np.array(points_3d, dtype=np.float64)
    colors_array = np.array(colors, dtype=np.uint8) if colors else None
    
    return points_array, colors_array


def find_colmap_files(colmap_path: Path, format_hint: Optional[str] = None) -> Path:
    """
    Поиск файла points3D в директории COLMAP
    """
    txt_file = colmap_path / 'points3D.txt'
    bin_file = colmap_path / 'points3D.bin'
    
    if format_hint == 'txt':
        if txt_file.exists():
            return txt_file
        else:
            raise FileNotFoundError(f"Файл {txt_file} не найден")
    elif format_hint == 'bin':
        if bin_file.exists():
            return bin_file
        else:
            raise FileNotFoundError(f"Файл {bin_file} не найден")
    
    # Автоматический выбор
    if txt_file.exists():
        return txt_file
    elif bin_file.exists():
        return bin_file
    else:
        raise FileNotFoundError(
            f"Не найден points3D.txt или points3D.bin в {colmap_path}"
        )


def save_results(
    points: np.ndarray,
    colors: Optional[np.ndarray],
    output_file: Path,
    with_colors: bool = False
) -> None:
    """
    Сохранение результатов в файл
    """
    if output_file.suffix == '.npz':
        if with_colors and colors is not None:
            np.savez(output_file, points=points, colors=colors)
        else:
            np.savez(output_file, points=points)
    elif output_file.suffix == '.npy':
        if with_colors and colors is not None:
            # Сохраняем как структурированный массив
            structured = np.zeros(
                len(points),
                dtype=[
                    ('x', 'f8'), ('y', 'f8'), ('z', 'f8'),
                    ('r', 'u1'), ('g', 'u1'), ('b', 'u1')
                ]
            )
            structured['x'] = points[:, 0]
            structured['y'] = points[:, 1]
            structured['z'] = points[:, 2]
            structured['r'] = colors[:, 0]
            structured['g'] = colors[:, 1]
            structured['b'] = colors[:, 2]
            np.save(output_file, structured)
        else:
            np.save(output_file, points)
    else:
        # CSV формат
        if with_colors and colors is not None:
            data = np.hstack([points, colors])
            header = 'x,y,z,r,g,b'
        else:
            data = points
            header = 'x,y,z'
        np.savetxt(output_file, data, delimiter=',', header=header, comments='')


def main():
    parser = argparse.ArgumentParser(
        description='Извлечение 3D точек из формата COLMAP',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры использования:
  python extract_points.py ./colmap_project
  python extract_points.py ./colmap_project points.npy --format bin
  python extract_points.py ./colmap_project points.npz --with-colors
  python extract_points.py ./colmap_project points.csv --with-colors
        """
    )
    
    parser.add_argument(
        'colmap_path',
        type=Path,
        help='Путь к директории с файлами COLMAP (sparse/0/ или dense/)'
    )
    
    parser.add_argument(
        'output_file',
        type=Path,
        nargs='?',
        default=Path('points.npy'),
        help='Выходной файл (по умолчанию: points.npy)'
    )
    
    parser.add_argument(
        '--format',
        choices=['txt', 'bin'],
        default=None,
        help='Формат файла COLMAP (txt или bin). Если не указан, определяется автоматически'
    )
    
    parser.add_argument(
        '--with-colors',
        action='store_true',
        help='Сохранить цвета точек (R, G, B) вместе с координатами'
    )
    
    parser.add_argument(
        '--stats',
        action='store_true',
        help='Вывести статистику по точкам'
    )
    
    args = parser.parse_args()
    
    # Проверяем существование директории
    if not args.colmap_path.exists():
        print(f"Ошибка: директория {args.colmap_path} не существует", file=sys.stderr)
        sys.exit(1)
    
    # Ищем файл points3D
    try:
        points_file = find_colmap_files(args.colmap_path, args.format)
        print(f"Найден файл: {points_file}")
    except FileNotFoundError as e:
        print(f"Ошибка: {e}", file=sys.stderr)
        sys.exit(1)
    
    # Парсим файл
    print("Парсинг файла...")
    try:
        if points_file.suffix == '.txt':
            points, colors = parse_colmap_txt(points_file)
        elif points_file.suffix == '.bin':
            points, colors = parse_colmap_bin(points_file)
        else:
            print(f"Ошибка: неизвестный формат файла {points_file.suffix}", file=sys.stderr)
            sys.exit(1)
    except Exception as e:
        print(f"Ошибка при парсинге: {e}", file=sys.stderr)
        sys.exit(1)
    
    # Выводим статистику
    if args.stats or True:  # Всегда выводим базовую статистику
        print(f"\nСтатистика:")
        print(f"  Количество точек: {len(points)}")
        print(f"  Диапазон X: [{points[:, 0].min():.3f}, {points[:, 0].max():.3f}]")
        print(f"  Диапазон Y: [{points[:, 1].min():.3f}, {points[:, 1].max():.3f}]")
        print(f"  Диапазон Z: [{points[:, 2].min():.3f}, {points[:, 2].max():.3f}]")
        
        if colors is not None:
            print(f"  Цвета: да (R, G, B)")
        
        # Вычисляем центр масс
        center = points.mean(axis=0)
        print(f"  Центр масс: [{center[0]:.3f}, {center[1]:.3f}, {center[2]:.3f}]")
    
    # Сохраняем результаты
    print(f"\nСохранение в {args.output_file}...")
    try:
        save_results(points, colors, args.output_file, args.with_colors)
        print(f"✓ Успешно сохранено {len(points)} точек")
    except Exception as e:
        print(f"Ошибка при сохранении: {e}", file=sys.stderr)
        sys.exit(1)
    
    # Пример использования результата
    print(f"\nПример использования в Python:")
    print(f"  import numpy as np")
    print(f"  data = np.load('{args.output_file}')")
    if args.output_file.suffix == '.npz':
        print(f"  points = data['points']")
        if args.with_colors:
            print(f"  colors = data['colors']")
    else:
        print(f"  points = data")


if __name__ == '__main__':
    main()