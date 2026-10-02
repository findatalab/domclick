#!/usr/bin/env python3
"""Merge Domclick XLSX exports and download linked images. Python 3.10+."""
import argparse
import csv
import hashlib
import json
import logging
import random
import re
import time
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from urllib.parse import urlsplit

import openpyxl
import requests
from PIL import Image

LOG = logging.getLogger('domclick')
CARD = re.compile(r'https://(?:[\w-]+\.)?domclick\.ru/card/(sale__[\w]+__(\d+))')
NUM = r'\d[\d\s\u00a0\u202f]*(?:[.,]\d+)?'


def number(value):
    match = re.search(NUM, str(value or ''))
    return float(re.sub(r'\s', '', match[0]).replace(',', '.')) if match else None


def image_key(url):
    """CDN variants of one original share the same /vitrina/... path."""
    p = urlsplit(url)
    return p.netloc + (p.path[p.path.index('/vitrina/'):] if '/vitrina/' in p.path else p.path)


def image_rank(url):
    m = re.search(r'/[cs](\d+)x(\d+)', url)
    return int(m[1]) * int(m[2]) if m else 0


def select_images(values):
    best = {}
    for value in values:
        url = str(value or '').strip()
        p = urlsplit(url)
        if p.scheme != 'https' or p.hostname != 'img.dmclk.ru':
            continue
        key = image_key(url)
        if key not in best or image_rank(url) > image_rank(best[key]):
            best[key] = url
    return list(best.values())


def parse_row(headers, values, source):
    raw = {f'{i}:{h}': v.isoformat() if isinstance(v, datetime) else v
           for i, (h, v) in enumerate(zip(headers, values)) if v is not None}
    pairs = [(str(h or '').strip(), str(v).strip()) for h, v in zip(headers, values) if v is not None]
    card = next((CARD.search(v) for _, v in pairs if CARD.search(v)), None)
    if not card:
        return None, raw
    def field(name):
        return next((v for h, v in pairs if h == name), None)
    def unit(pattern):
        # Only card summary fields: do not confuse narrative with advertised area.
        return next((v for h, v in pairs if h.startswith('s6KKu') and re.search(pattern, v, re.I)), None)
    price = field('s5oAg') or next((v for _, v in pairs if re.fullmatch(r'[\d\s.,]+\s*₽', v)), None)
    area = number(unit(r'м[²2]'))
    floors = unit(r'\d+\s*-?\s*эт')
    tags = [v for h, v in pairs if h == 'sa2uH' or h.startswith('sa2uH ')]
    badges = [v for h, v in pairs if h.startswith('bdg-text-')]
    land_use = [v for h, v in pairs if h.startswith('s6KKu') and v.upper() in {'ИЖС', 'ЛПХ', 'СНТ', 'ДНП'}]
    record = {
        'listing_id': card[2], 'listing_url': card[0], 'object_type': card[1].rsplit('__', 1)[0],
        'price_type': 'asking_price', 'currency': 'RUB', 'asking_price_rub': number(price),
        'price_per_m2_advertised': number(field('sa3w3')),
        'house_area_m2': area, 'land_area_sotka': number(unit(r'сот')),
        'floors': int(re.search(r'\d+', floors)[0]) if floors else None,
        'address_fragment': field('shM1H'), 'locality': field('sBVaF 2'),
        'locality_url': field('sBVaF href 2'), 'settlement': field('link-link-777-13-1-6'),
        'highway': field('s0nV9'), 'distance_to_mkad_km': number(field('smKxN')),
        'description': field('sK4aL'), 'seller_name': field('sPeBt'), 'seller_type': field('sx6J2'),
        'listing_age_text': field('sMekc'), 'land_use_label': land_use,
        'tags_seller_claimed': tags, 'badges_platform_claimed': badges,
        'photo_count_advertised': number((field('gallery-text-bf2-24-1-0') or '').split('из')[-1]),
        'photo_urls': select_images(values), 'sources': [source], 'conflicts': [],
    }
    record['price_per_m2_computed'] = round(record['asking_price_rub'] / area, 2) if area and record['asking_price_rub'] else None
    return record, raw


def write_json(path, value):
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    temp.replace(path)


def collect(folder, out, recursive=False):
    objects, skipped, files = {}, 0, 0
    paths = sorted(folder.rglob('*.xlsx') if recursive else folder.glob('*.xlsx'))
    with (out / 'raw_rows.jsonl').open('w', encoding='utf-8') as raw_file:
        for path in paths:
            if path.name.startswith('~$'):
                continue
            try:
                book = openpyxl.load_workbook(path, read_only=True, data_only=True)
            except Exception as exc:
                LOG.error('Cannot read %s: %s', path, exc)
                continue
            files += 1
            try:
                for sheet in book:
                    rows = sheet.iter_rows(values_only=True)
                    headers = next(rows, ())
                    for row_no, values in enumerate(rows, 2):
                        if not any(v is not None for v in values):
                            continue
                        source = {'file': str(path.relative_to(folder)), 'sheet': sheet.title, 'row': row_no}
                        record, raw = parse_row(headers, values, source)
                        raw_file.write(json.dumps({'source': source, 'listing_id': record['listing_id'] if record else None,
                                                   'cells': raw}, ensure_ascii=False, default=str) + '\n')
                        if record is None:
                            skipped += 1
                            LOG.warning('No listing URL: %s', source)
                            continue
                        key = record['listing_id']
                        if key not in objects:
                            objects[key] = record
                            continue
                        old = objects[key]
                        old['sources'].append(source)
                        for k, value in record.items():
                            if k in {'sources', 'conflicts', 'photo_urls'}:
                                continue
                            if old[k] in (None, '', []):
                                old[k] = value
                            elif value not in (None, '', []) and old[k] != value:
                                old['conflicts'].append({'field': k, 'kept': old[k], 'other': value, 'source': source})
                        old['photo_urls'] = select_images(old['photo_urls'] + record['photo_urls'])
            finally:
                book.close()
    return list(objects.values()), {'files_read': files, 'rows_without_listing': skipped}


def retry_after(value):
    if not value:
        return None
    try:
        return max(0, float(value))
    except ValueError:
        try:
            return max(0, (parsedate_to_datetime(value) - datetime.now(timezone.utc)).total_seconds())
        except (ValueError, TypeError):
            return None


class Blocked(Exception):
    pass


class Downloader:
    def __init__(self, args):
        self.args = args
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': args.user_agent,
            'Accept': 'image/webp,image/jpeg,image/png,image/*;q=0.8',
            'Accept-Language': 'ru-RU,ru;q=0.9,en;q=0.5',
        })

    def download(self, url, dest):
        p = urlsplit(url)
        if p.scheme != 'https' or p.hostname != 'img.dmclk.ru':
            return {'status': 'invalid_url'}
        if dest.exists():
            try:
                return self.inspect(dest, 'cached')
            except Exception:
                dest.unlink()
        temp = dest.with_suffix(dest.suffix + '.part')
        try:
            for attempt in range(self.args.retries + 1):
                time.sleep(random.uniform(self.args.delay_min, self.args.delay_max))
                try:
                    # Referrer is not fabricated: direct download of exported image URL.
                    # Redirects disabled to avoid silently leaving the known image host.
                    with self.session.get(url, stream=True, timeout=(10, 45), allow_redirects=False) as response:
                        status = response.status_code
                        if status in (401, 403):
                            raise Blocked(f'HTTP {status}: stop downloads; data will be saved')
                        if status == 429 or status in (500, 502, 503, 504):
                            wait = retry_after(response.headers.get('Retry-After'))
                            wait = wait if wait is not None else min(300, 15 * 2 ** attempt) + random.uniform(0, 5)
                            if wait > self.args.max_wait:
                                raise Blocked(f'Server requests {wait:.0f}s wait; rerun later')
                            if attempt == self.args.retries:
                                if status == 429:
                                    raise Blocked('Persistent HTTP 429: stop downloads')
                                return {'status': 'failed', 'error': f'HTTP {status}'}
                            LOG.warning('HTTP %s, waiting %.1fs', status, wait)
                            time.sleep(wait)
                            continue
                        if status != 200:
                            return {'status': 'failed', 'error': f'HTTP {status}'}
                        if not response.headers.get('Content-Type', '').lower().startswith('image/'):
                            return {'status': 'failed', 'error': 'Response is not an image'}
                        size = 0
                        with temp.open('wb') as output:
                            for chunk in response.iter_content(65536):
                                size += len(chunk)
                                if size > self.args.max_image_mb * 1024 * 1024:
                                    raise ValueError('Image size limit exceeded')
                                output.write(chunk)
                        info = self.inspect(temp, 'downloaded')
                        temp.replace(dest)
                        return info
                except requests.RequestException as exc:
                    if attempt == self.args.retries:
                        return {'status': 'failed', 'error': str(exc)}
                    time.sleep(min(60, 2 ** (attempt + 2)) + random.uniform(0, 3))
                except (ValueError, OSError, Image.DecompressionBombError) as exc:
                    return {'status': 'failed', 'error': str(exc)}
        finally:
            temp.unlink(missing_ok=True)

    @staticmethod
    def inspect(path, status):
        with Image.open(path) as image:
            width, height = image.size
            fmt = image.format
            image.verify()
        sha = hashlib.sha256(path.read_bytes()).hexdigest()
        return {'status': status, 'width': width, 'height': height, 'format': fmt,
                'bytes': path.stat().st_size, 'sha256': sha}


def save_dataset(out, records):
    with (out / 'objects.jsonl.tmp').open('w', encoding='utf-8') as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')
    (out / 'objects.jsonl.tmp').replace(out / 'objects.jsonl')
    if not records:
        return
    with (out / 'objects.csv.tmp').open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0]), delimiter=';')
        writer.writeheader()
        for record in records:
            writer.writerow({k: json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict)) else v
                             for k, v in record.items()})
    (out / 'objects.csv.tmp').replace(out / 'objects.csv')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', type=Path, help='Directory with XLSX files')
    parser.add_argument('--output', type=Path, default=Path('dataset'))
    parser.add_argument('--recursive', action='store_true')
    parser.add_argument('--no-download', action='store_true')
    parser.add_argument('--delay-min', type=float, default=2)
    parser.add_argument('--delay-max', type=float, default=5)
    parser.add_argument('--retries', type=int, default=3, help='Additional attempts')
    parser.add_argument('--max-wait', type=float, default=900)
    parser.add_argument('--max-image-mb', type=float, default=30)
    parser.add_argument('--max-photos', type=int, default=0, help='Per listing; 0 = all')
    parser.add_argument('--user-agent', default='DomclickResearchDataset/1.0 (Python requests; image download)')
    args = parser.parse_args()
    if not args.input.is_dir():
        parser.error('Input directory does not exist')
    if not 0 <= args.delay_min <= args.delay_max or args.retries < 0 or args.max_photos < 0 or args.max_image_mb <= 0 or args.max_wait < 0:
        parser.error('Invalid limits or delays')
    args.output.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s',
                        handlers=[logging.StreamHandler(), logging.FileHandler(args.output / 'run.log', encoding='utf-8')])
    records, summary = collect(args.input, args.output, args.recursive)
    previous = {}
    if (args.output / 'objects.jsonl').exists():
        for line in (args.output / 'objects.jsonl').read_text(encoding='utf-8').splitlines():
            item = json.loads(line)
            previous[item['listing_id']] = item.get('photo_paths', [])
    for record in records:
        record['photo_paths'] = [p for p in previous.get(record['listing_id'], [])
                                 if (args.output / p).is_file()]
        area, price = record['house_area_m2'], record['asking_price_rub']
        record['price_per_m2_computed'] = round(price / area, 2) if area and price else None
    save_dataset(args.output, records)  # Dataset is available before network work.
    manifest_path = args.output / 'photos_manifest.jsonl'
    downloader = Downloader(args)
    blocked = False
    attempts = 0
    try:
        with manifest_path.open('a', encoding='utf-8') as manifest:
            for i, record in enumerate(records, 1):
                LOG.info('Listing %s/%s: %s, %s images', i, len(records), record['listing_id'], len(record['photo_urls']))
                urls = record['photo_urls'][:args.max_photos] if args.max_photos else record['photo_urls']
                if not args.no_download:
                    photo_dir = args.output / 'photos' / record['listing_id']
                    photo_dir.mkdir(parents=True, exist_ok=True)
                    for url in urls:
                        name = hashlib.sha256(image_key(url).encode()).hexdigest()[:24] + (Path(urlsplit(url).path).suffix or '.img')
                        path = photo_dir / name
                        info = downloader.download(url, path)
                        attempts += 1
                        relative = path.relative_to(args.output).as_posix()
                        manifest.write(json.dumps({'listing_id': record['listing_id'], 'url': url, 'path': relative,
                                                   'processed_at': datetime.now(timezone.utc).isoformat(), **info}, ensure_ascii=False) + '\n')
                        manifest.flush()
                        if info['status'] in ('cached', 'downloaded'):
                            if relative not in record['photo_paths']:
                                record['photo_paths'].append(relative)
                        else:
                            LOG.warning('Image failed: %s', info)
                save_dataset(args.output, records)  # Checkpoint after every listing.
    except (Blocked, KeyboardInterrupt) as exc:
        blocked = True
        LOG.warning('Stopped: %s', str(exc) or 'Keyboard interrupt')
    finally:
        downloader.session.close()
        save_dataset(args.output, records)
        summary.update({'objects': len(records), 'download_attempts_in_run': attempts,
                        'objects_with_conflicts': sum(bool(r['conflicts']) for r in records),
                        'photo_links': sum(len(r['photo_urls']) for r in records),
                        'photo_paths_in_run': sum(len(r['photo_paths']) for r in records),
                        'downloads_stopped': blocked, 'processed_at': datetime.now(timezone.utc).isoformat()})
        write_json(args.output / 'summary.json', summary)
        LOG.info('Result: %s', summary)
    return 2 if blocked else 0


if __name__ == '__main__':
    raise SystemExit(main())
