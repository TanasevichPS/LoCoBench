#!/usr/bin/env python3
"""
Скрипт для проверки отсутствующих файлов в сценариях.
Проходит по директории data/generated и сравнивает файлы с файлами из сценариев.
"""

import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple


def clean_project_path(project_path: str) -> Optional[str]:
    """Нормализовать путь до проекта относительно data/generated."""
    if not project_path:
        return None

    normalized = str(project_path).strip()
    if not normalized:
        return None

    normalized = normalized.replace('\\', '/')
    while normalized.startswith('./'):
        normalized = normalized[2:]
    normalized = normalized.lstrip('/')

    prefixes = (
        'data/generated/',
        './data/generated/',
        'generated/',
        './generated/',
    )
    for prefix in prefixes:
        if normalized.startswith(prefix):
            normalized = normalized[len(prefix):]

    normalized = normalized.strip('/')
    return normalized or None


def contains_project_number(project_key: str) -> bool:
    """Проверить, содержит ли имя проекта числовой идентификатор (в последнем сегменте)."""
    if not project_key:
        return False
    last_segment = project_key.replace('\\', '/').split('/')[-1]
    return any(ch.isdigit() for ch in last_segment)


def parse_project_from_scenario_id(scenario_id: str) -> Optional[str]:
    """Попробовать извлечь имя проекта из идентификатора сценария."""
    if not scenario_id:
        return None

    parts = [p for p in scenario_id.split('_') if p]
    if len(parts) < 2:
        return None

    for idx, part in enumerate(parts):
        if part.isdigit() and len(part) >= 3:
            candidate = '_'.join(parts[: idx + 1])
            if candidate:
                return candidate

    return None


def extract_project_identifier(scenario: Dict) -> Tuple[Optional[str], Optional[str]]:
    """Вернуть нормализованный идентификатор проекта и источник данных."""
    metadata = scenario.get('metadata') or {}
    for key in ['project_path', 'project_root', 'project_dir', 'project_directory']:
        candidate = metadata.get(key)
        cleaned = clean_project_path(candidate)
        if cleaned:
            return cleaned, f"metadata.{key}"

    scenario_keys = [
        'project_path',
        'project_root',
        'project_dir',
        'project_directory',
        'project_name',
        'projectId',
        'project_id',
    ]
    for key in scenario_keys:
        candidate = scenario.get(key)
        cleaned = clean_project_path(candidate)
        if cleaned:
            return cleaned, f"scenario.{key}"

    candidate = parse_project_from_scenario_id(scenario.get('id', ''))
    if candidate:
        return candidate, 'scenario.id'

    return None, None


def extract_context_file_paths(scenario: Dict) -> List[str]:
    """Получить список путей контекстных файлов из сценария."""
    paths: List[str] = []

    context_files = scenario.get('context_files', [])
    if isinstance(context_files, dict):
        paths.extend([p for p in context_files.keys() if isinstance(p, str)])
    elif isinstance(context_files, list):
        paths.extend([p for p in context_files if isinstance(p, str)])
    elif isinstance(context_files, str):
        paths.append(context_files)

    metadata_context = scenario.get('metadata', {}).get('context_files')
    if isinstance(metadata_context, dict):
        paths.extend([p for p in metadata_context.keys() if isinstance(p, str)])
    elif isinstance(metadata_context, list):
        paths.extend([p for p in metadata_context if isinstance(p, str)])

    return [p for p in paths if p]


def find_scenario_files(generated_dir: Path, scenarios_dir: Path) -> Dict[str, List[str]]:
    """Найти все сценарии и извлечь из них список файлов по проектам."""
    project_files: Dict[str, Set[str]] = defaultdict(set)
    skipped_no_project = 0
    skipped_no_number = 0
    skipped_empty_context = 0
    total_scenarios = 0

    scenario_paths = sorted(scenarios_dir.glob('*.json'))
    if not scenario_paths:
        print(f"⚠️ В директории {scenarios_dir} не найдено json-файлов сценариев")

    for scenario_path in scenario_paths:
        total_scenarios += 1
        try:
            with scenario_path.open('r', encoding='utf-8') as f:
                scenario = json.load(f)
        except Exception as exc:
            print(f"Ошибка при чтении {scenario_path}: {exc}")
            continue

        project_name, source = extract_project_identifier(scenario)
        scenario_id = scenario.get('id', '<без id>')

        if not project_name:
            skipped_no_project += 1
            print(f"⚠️ Пропускаем сценарий {scenario_path.name} (id={scenario_id}): не удалось определить проект.")
            continue

        if not contains_project_number(project_name):
            skipped_no_number += 1
            print(
                f"⚠️ Пропускаем сценарий {scenario_path.name} (id={scenario_id}): "
                f"имя проекта '{project_name}' не содержит числового идентификатора."
            )
            continue

        context_paths = extract_context_file_paths(scenario)
        if not context_paths:
            skipped_empty_context += 1
            print(f"⚠️ Сценарий {scenario_path.name} (id={scenario_id}): список context_files пуст.")
            continue

        project_files[project_name].update(context_paths)

    project_files_cleaned = {project: sorted(paths) for project, paths in project_files.items()}

    print(f"\nВсего сценариев: {total_scenarios}")
    print(f"Проектов после фильтрации: {len(project_files_cleaned)}")
    if skipped_no_project:
        print(f"Пропущено сценариев без указания проекта: {skipped_no_project}")
    if skipped_no_number:
        print(f"Пропущено сценариев без номера в имени проекта: {skipped_no_number}")
    if skipped_empty_context:
        print(f"Пропущено сценариев с пустыми context_files: {skipped_empty_context}")

    if project_files_cleaned:
        print('Примеры проектов:')
        for index, (project_name, files) in enumerate(list(project_files_cleaned.items())[:20], 1):
            print(f"  {index}. {project_name} ({len(files)} файлов)")

    return project_files_cleaned


def find_actual_files(generated_dir: Path, project_name: str) -> Set[str]:
    """Найти все реальные файлы в директории проекта."""
    actual_files: Set[str] = set()
    project_path = generated_dir / project_name

    if not project_path.exists():
        return actual_files

    for root, _, files in os.walk(project_path):
        root_path = Path(root)
        rel_root = root_path.relative_to(project_path)

        for file in files:
            if rel_root == Path('.'):
                rel_path = Path(file)
            else:
                rel_path = rel_root / file
            actual_files.add(str(rel_path).replace('\\', '/'))

    return actual_files


def normalize_expected_path(path: str, project_name: str) -> str:
    """Нормализовать ожидаемый путь файла относительно корня проекта."""
    if not path:
        return ''

    normalized = str(path).replace('\\', '/').strip()
    if not normalized:
        return ''

    while normalized.startswith('./'):
        normalized = normalized[2:]
    normalized = normalized.lstrip('/')

    prefixes = (
        'data/generated/',
        './data/generated/',
        'generated/',
        './generated/',
    )
    for prefix in prefixes:
        if normalized.startswith(prefix):
            normalized = normalized[len(prefix):]

    normalized = normalized.strip('/')
    if not normalized:
        return ''

    components = [part for part in normalized.split('/') if part and part != '.']
    while components and components[0] in {'data', 'generated'}:
        components.pop(0)

    project_parts = [part for part in project_name.replace('\\', '/').split('/') if part]
    if project_parts:
        while len(components) >= len(project_parts) and components[: len(project_parts)] == project_parts:
            components = components[len(project_parts):]
        last_part = project_parts[-1]
        while components and components[0] == last_part:
            components = components[1:]

    normalized_path = '/'.join(components)
    if not normalized_path:
        normalized_path = Path(path).name

    return normalized_path.replace('\\', '/')


def check_missing_files(generated_dir: Path, scenarios_dir: Path) -> Dict[str, Dict]:
    """Проверить отсутствующие файлы для каждого проекта."""
    results: Dict[str, Dict] = {}

    scenario_files = find_scenario_files(generated_dir, scenarios_dir)
    print(f"Найдено {len(scenario_files)} проектов в сценариях")

    for project_name, expected_files in scenario_files.items():
        print(f"\nПроверка проекта: {project_name}")
        print(f"  Ожидается файлов: {len(expected_files)}")

        project_path = generated_dir / project_name
        actual_files = find_actual_files(generated_dir, project_name)
        print(f"  Найдено файлов: {len(actual_files)}")

        if not project_path.exists():
            print(f"  ⚠️ Директория {project_name} не найдена!")
            results[project_name] = {
                'total_expected': len(expected_files),
                'total_found': 0,
                'total_missing': len(expected_files),
                'missing_files': [{'expected_path': f, 'normalized_path': f} for f in expected_files[:50]],
                'missing_count': len(expected_files),
                'error': 'Project directory not found',
            }
            continue

        missing_files: List[Dict] = []
        found_files: List[str] = []

        for expected_file in expected_files:
            normalized_expected = normalize_expected_path(expected_file, project_name)

            path_candidates = []
            if normalized_expected:
                path_candidates.append(normalized_expected)
                path_candidates.append(normalized_expected.lstrip('/'))

            original_clean = str(expected_file).replace('\\', '/')
            if original_clean:
                path_candidates.append(original_clean)
                path_candidates.append(original_clean.lstrip('/'))

            seen: Set[str] = set()
            path_attempts: List[str] = []
            for candidate in path_candidates:
                candidate = candidate.strip()
                if not candidate:
                    continue
                candidate = candidate.replace('\\', '/')
                if candidate not in seen:
                    seen.add(candidate)
                    path_attempts.append(candidate)

            if normalized_expected:
                path_parts = [part for part in normalized_expected.split('/') if part]
                if len(path_parts) >= 2:
                    for suffix_len in range(2, min(len(path_parts), 4) + 1):
                        suffix = '/'.join(path_parts[-suffix_len:])
                        if suffix not in seen:
                            seen.add(suffix)
                            path_attempts.append(suffix)

            found = False
            for attempt in path_attempts:
                if attempt in actual_files:
                    found = True
                    break

            if not found:
                for attempt in path_attempts:
                    potential_file = project_path / attempt
                    if potential_file.exists() and potential_file.is_file():
                        rel_path = potential_file.relative_to(project_path)
                        rel_path_str = str(rel_path).replace('\\', '/')
                        actual_files.add(rel_path_str)
                        found = True
                        break

            if not found and normalized_expected:
                suffix_variants = [normalized_expected, normalized_expected.lstrip('/')]
                for suffix in suffix_variants:
                    matches = [p for p in actual_files if p.endswith(suffix)]
                    if len(matches) == 1:
                        found = True
                        break

            if not found:
                file_name = Path(expected_file).name
                if file_name:
                    matches_found = []
                    for root, _, files in os.walk(project_path):
                        if file_name in files:
                            matches_found.append(Path(root) / file_name)

                    if len(matches_found) == 1:
                        match_rel = matches_found[0].relative_to(project_path)
                        actual_files.add(str(match_rel).replace('\\', '/'))
                        found = True
                    elif len(matches_found) > 1 and normalized_expected:
                        for match in matches_found:
                            match_rel = match.relative_to(project_path)
                            match_str = str(match_rel).replace('\\', '/')
                            if match_str.endswith(normalized_expected):
                                actual_files.add(match_str)
                                found = True
                                break

            if found:
                found_files.append(expected_file)
            else:
                missing_files.append({
                    'expected_path': expected_file,
                    'normalized_path': normalized_expected,
                    'attempted_paths': path_attempts[:5],
                })

        results[project_name] = {
            'total_expected': len(expected_files),
            'total_found': len(found_files),
            'total_missing': len(missing_files),
            'missing_files': missing_files[:50],
            'missing_count': len(missing_files),
            'found_files_count': len(found_files),
        }

        print(f"  Найдено: {len(found_files)}, Отсутствует: {len(missing_files)}")
        if len(found_files) == 0 and len(expected_files) > 0:
            print(
                f"  ⚠️ ВНИМАНИЕ: Для проекта {project_name} не найдено ни одного файла из {len(expected_files)} ожидаемых!"
            )
            print(f"     Примеры ожидаемых путей: {expected_files[:3]}")
            print(f"     Примеры реальных путей в директории: {list(actual_files)[:5]}")

    return results


def main():
    generated_dir = Path('data/generated')
    scenarios_dir = Path('data/output/scenarios')
    output_file = Path('missing_files_report.json')

    if not generated_dir.exists():
        print(f"Директория {generated_dir} не найдена!")
        return

    if not scenarios_dir.exists():
        print(f"Директория {scenarios_dir} не найдена!")
        return

    print('Начало проверки файлов...')
    results = check_missing_files(generated_dir, scenarios_dir)

    with output_file.open('w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\nРезультаты сохранены в {output_file}")

    total_missing = sum(r['missing_count'] for r in results.values())
    total_expected = sum(r['total_expected'] for r in results.values())
    total_found = sum(r['total_found'] for r in results.values())

    print('\nОбщая статистика:')
    print(f"  Проектов проверено: {len(results)}")
    print(f"  Всего ожидалось файлов: {total_expected}")
    print(f"  Всего найдено файлов: {total_found}")
    print(f"  Всего отсутствует файлов: {total_missing}")

    sorted_projects = sorted(results.items(), key=lambda x: x[1]['missing_count'], reverse=True)
    print('\nТоп-10 проектов с наибольшим количеством отсутствующих файлов:')
    for idx, (project, data) in enumerate(sorted_projects[:10], 1):
        print(
            f"  {idx}. {project}: {data['missing_count']}/{data['total_expected']} отсутствует (найдено: {data['total_found']})"
        )


if __name__ == '__main__':
    main()
