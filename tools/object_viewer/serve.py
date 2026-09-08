"""Run from any directory: python3 tools/object_viewer/serve.py [--port 8765]."""
import argparse
from dataclasses import asdict
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import re
import sys
from urllib.parse import unquote, urlparse

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from gentle_manip.assets.registry import OBJECT_MAP

FOOD = {'tofu', 'mushroom', 'cherry_tomato', 'tomato', 'banana_chunk',
        'pasta_bundle', 'raspberry', 'banana', 'strawberry'}

# Names registered by the object-expansion pipeline (source_and_filter -> mesh_prep ->
# register_object) are recorded here so the atlas can group them separately from the
# hand-curated food/dev sets; see gentle_manip/scripts/object_expansion/EXPANSION_LOG.csv.
EXPANSION_LOG = ROOT / 'gentle_manip' / 'scripts' / 'object_expansion' / 'EXPANSION_LOG.csv'

def expansion_names():
    if not EXPANSION_LOG.exists():
        return {}
    import csv
    out = {}
    with open(EXPANSION_LOG) as f:
        for row in csv.DictReader(f):
            out[row['name']] = row
    return out

def catalog():
    items = []
    exp = expansion_names()
    for name, obj in OBJECT_MAP.items():
        category = re.sub(r'\d+$', '', name)
        if name in exp:
            group = 'Object expansion'
        elif category in FOOD:
            group = 'Food'
        else:
            group = 'Development / synthetic'
        path = Path(obj.mesh_path) if obj.mesh_path else None
        items.append(dict(name=name, category=category, group=group,
            mesh='/' + path.relative_to(ROOT).as_posix() if path and path.exists() else None,
            missing=bool(path and not path.exists()), size=obj.size,
            material=asdict(obj.material), object_type=obj.object_type,
            parked=category == 'raspberry',
            expansion_meta=exp.get(name)))
    return items

def find_runs(name: str, limit: int = 6):
    """Most recent collect_demos_synth_v4 run dirs for single_lift_<name>_soft, with stats
    and served-relative paths to videos / final-grasp PNGs."""
    import yaml
    base = ROOT / 'dataset' / 'demos' / f'single_lift_{name}_soft'
    if not base.exists():
        return []
    run_dirs = sorted([d for d in base.iterdir() if d.is_dir()], key=lambda d: d.name, reverse=True)
    out = []
    for run_dir in run_dirs[:limit]:
        stats_path = run_dir / 'stats.yaml'
        stats = yaml.safe_load(stats_path.read_text()) if stats_path.exists() else None
        vids, pngs = [], []
        vdir = run_dir / 'videos'
        if vdir.exists():
            for f in sorted(vdir.iterdir()):
                rel = '/' + f.relative_to(ROOT).as_posix()
                (vids if f.suffix == '.mp4' else pngs if f.suffix == '.png' else []).append(rel) \
                    if f.suffix in ('.mp4', '.png') else None
        out.append(dict(run_dir=run_dir.name, stats=stats, videos=vids[:8], images=pngs[:8]))
    return out

CANDIDATE_CSVS = [
    ROOT / 'dataset' / 'object_expansion' / 'candidates_all.csv',
    ROOT / 'dataset' / 'object_expansion' / 'candidates.csv',
    ROOT / 'dataset' / 'object_expansion' / 'candidates_metafood3d.csv',
    ROOT / 'dataset' / 'object_expansion' / 'candidates_thinshell_retry.csv',
] + sorted(ROOT.glob('dataset/object_expansion/candidates_broad_*.csv'))

def candidates():
    """Sourced (pre-registration) candidates from every candidates*.csv the pipeline has
    written -- raw .glb/.obj files straight from the source dataset, geometry-filter verdict
    only (mesh_prep/FEM-gate/registration have NOT run yet). Servable path is the file's
    location relative to ROOT (works through the objaverse_cache symlink to /nobackup too)."""
    import csv
    seen, out = set(), []
    for csv_path in CANDIDATE_CSVS:
        if not csv_path.exists():
            continue
        with open(csv_path) as f:
            for row in csv.DictReader(f):
                key = (row.get('category'), row.get('uid'))
                if key in seen or row.get('verdict') not in ('accept', 'borderline'):
                    continue
                seen.add(key)
                local_path = row.get('local_path') or ''
                mesh_url = None
                try:
                    # NOT .resolve() -- dataset/object_expansion/objaverse_cache is a symlink
                    # out to /nobackup (disk-quota fix, see DEVLOG); resolving it walks OUTSIDE
                    # ROOT and breaks relative_to. The stored local_path is already absolute and
                    # already under ROOT (via the symlink component), so use it as-is -- the OS
                    # follows the symlink transparently when the file is actually opened to serve.
                    p = Path(local_path)
                    mesh_url = '/' + p.relative_to(ROOT).as_posix()
                except Exception:
                    mesh_url = None
                out.append(dict(
                    uid=row.get('uid'), category=row.get('category'), bucket=row.get('bucket'),
                    source=row.get('source') or 'objaverse', verdict=row.get('verdict'),
                    mesh=mesh_url, ext=Path(local_path).suffix.lower(),
                    min_width_mm=row.get('min_width_scaled_mm'),
                    max_extent_mm=row.get('max_extent_scaled_mm'),
                    thin_axis_mm=row.get('thin_axis_scaled_mm'),
                    suggested_scale=row.get('suggested_scale'), reason=row.get('reason'),
                    from_csv=csv_path.name))
    out.sort(key=lambda r: (r['category'], r['uid'] or ''))
    return out

class Handler(SimpleHTTPRequestHandler):
    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == '/api/candidates':
            data = json.dumps(candidates()).encode()
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(data)))
            self.end_headers()
            self.wfile.write(data)
        elif parsed.path == '/api/objects':
            data = json.dumps(catalog()).encode()
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(data)))
            self.end_headers()
            self.wfile.write(data)
        elif parsed.path.startswith('/api/runs/'):
            name = unquote(parsed.path[len('/api/runs/'):])
            data = json.dumps(find_runs(name)).encode()
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(data)))
            self.end_headers()
            self.wfile.write(data)
        elif self.path == '/':
            self.send_response(302)
            self.send_header('Location', '/tools/object_viewer/index.html')
            self.end_headers()
        else:
            super().do_GET()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, default=8765)
    args = parser.parse_args()
    print(f'Object viewer: http://localhost:{args.port}', flush=True)
    ThreadingHTTPServer(('127.0.0.1', args.port), partial(Handler, directory=str(ROOT))).serve_forever()
