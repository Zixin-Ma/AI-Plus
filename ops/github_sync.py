#!/usr/bin/python3
"""Independent, conservative GitHub synchronization for one server worktree."""
import argparse
import datetime
import fcntl
import json
import logging
from logging.handlers import RotatingFileHandler
import os
from pathlib import Path
import re
import signal
import subprocess
import time

# Keep credentials and runtime state outside the public source repository.
HERE = Path('/opt/data/private/mzx/.bjtu-remote-access/ai-plus-sync')
LOG = logging.getLogger('ai-plus-sync')
MAX_BYTES = 50 * 1024 * 1024
SECRET = re.compile(rb'(?:gh[pousr]_[A-Za-z0-9]{30,255}|github_pat_[A-Za-z0-9_]{50,255}|AKIA[A-Z0-9]{16}|-----BEGIN (?:OPENSSH |RSA |EC |DSA |ENCRYPTED )?PRIVATE KEY-----)')


def now():
    return datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat()


def save(path, value):
    tmp = path.with_name(path.name + '.new')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    os.chmod(str(tmp), 0o600)
    os.replace(str(tmp), str(path))


def git(repo, *args, check=True, binary=False):
    result = subprocess.run(
        ['git', '-C', str(repo)] + list(args), capture_output=True,
        text=not binary, timeout=180,
        env={**os.environ, 'GIT_TERMINAL_PROMPT': '0', 'GIT_MERGE_AUTOEDIT': 'no'},
    )
    if check and result.returncode:
        message = result.stderr.decode(errors='replace') if binary else result.stderr
        raise RuntimeError('git ' + args[0] + ': ' + message[-1600:].strip())
    return result


def clean(repo):
    return not git(repo, 'status', '--porcelain', '--untracked-files=normal').stdout.strip()


def audit_index(repo):
    entries = git(repo, 'ls-files', '--stage', '-z', binary=True).stdout.split(b'\0')
    seen = set()
    for entry in entries:
        if not entry:
            continue
        metadata, name = entry.split(b'\t', 1)
        mode, oid, stage = metadata.split()
        label = os.fsdecode(name)
        if stage != b'0' or mode == b'160000':
            raise RuntimeError('Unmerged entry or nested repository requires review: ' + label)
        if oid in seen:
            continue
        seen.add(oid)
        size = int(git(repo, 'cat-file', '-s', oid.decode()).stdout)
        if size > MAX_BYTES:
            raise RuntimeError('File exceeds 50 MiB; add a storage policy before syncing: ' + label)
        content = git(repo, 'cat-file', 'blob', oid.decode(), binary=True).stdout
        if SECRET.search(content):
            raise RuntimeError('Possible credential detected; value withheld: ' + label)


def audit_outgoing(repo, ref):
    # Also inspect manually created commits, including blobs deleted in later commits.
    objects = git(repo, 'rev-list', '--objects', ref + '..HEAD').stdout.splitlines()
    for row in objects:
        oid = row.split(' ', 1)[0]
        if git(repo, 'cat-file', '-t', oid).stdout.strip() != 'blob':
            continue
        if int(git(repo, 'cat-file', '-s', oid).stdout) > MAX_BYTES:
            raise RuntimeError('Outgoing history contains an oversize blob; review required')
        if SECRET.search(git(repo, 'cat-file', 'blob', oid, binary=True).stdout):
            raise RuntimeError('Outgoing history contains a possible credential; value withheld')


def synchronize(config):
    repo = Path(config['repo']).resolve()
    branch = config.get('branch', 'main')
    remote = config['remote']
    if git(repo, 'rev-parse', '--show-toplevel').stdout.strip() != str(repo):
        raise RuntimeError('Worktree is not the expected independent repository')
    if git(repo, 'remote', 'get-url', 'origin').stdout.strip() != remote:
        raise RuntimeError('Origin changed; automatic synchronization stopped')
    if git(repo, 'symbolic-ref', '--quiet', '--short', 'HEAD').stdout.strip() != branch:
        raise RuntimeError('Branch changed; automatic synchronization stopped')
    gitdir = Path(git(repo, 'rev-parse', '--absolute-git-dir').stdout.strip())
    if any((gitdir / name).exists() for name in
           ['MERGE_HEAD', 'CHERRY_PICK_HEAD', 'REVERT_HEAD', 'rebase-merge', 'rebase-apply']):
        raise RuntimeError('Merge/rebase in progress; resolve it before synchronization resumes')
    if git(repo, 'ls-files', '-u').stdout:
        raise RuntimeError('Unresolved conflict; automatic synchronization paused')

    # Save working changes before integrating remote history. Never stash or reset.
    git(repo, 'add', '-A')
    staged = git(repo, 'diff', '--cached', '--quiet', check=False).returncode
    if staged not in (0, 1):
        raise RuntimeError('Could not inspect staged changes')
    committed = False
    if staged:
        audit_index(repo)
        time.sleep(config.get('settle_seconds', 2))
        if git(repo, 'diff', '--quiet', check=False).returncode:
            raise RuntimeError('Files are still changing; retry on the next cycle')
        git(repo, 'commit', '-m', 'sync: BJTU changes ' + now())
        committed = True
    if not clean(repo):
        raise RuntimeError('New edits arrived during snapshot; retry on the next cycle')
    ref = 'refs/remotes/origin/' + branch
    for attempt in range(3):
        git(repo, 'fetch', '--no-tags', 'origin', 'refs/heads/' + branch + ':' + ref)
        if not clean(repo):
            raise RuntimeError('Working files changed during fetch; integration deferred')
        result = git(repo, 'merge', '--no-edit', ref, check=False)
        if result.returncode:
            raise RuntimeError('Remote merge requires review; both commits retained. ' +
                               (result.stdout + result.stderr)[-1200:])
        if not clean(repo):
            raise RuntimeError('Working files changed during integration; push deferred')
        ahead = int(git(repo, 'rev-list', '--count', ref + '..HEAD').stdout)
        if ahead:
            audit_index(repo)
            audit_outgoing(repo, ref)
            result = git(repo, 'push', '--porcelain', 'origin', 'HEAD:refs/heads/' + branch, check=False)
            if result.returncode:
                if attempt < 2 and any(x in result.stderr.lower() for x in ['rejected', 'fetch first', 'non-fast-forward']):
                    continue
                raise RuntimeError('Push failed; local commits retained: ' + result.stderr[-1200:])
        head = git(repo, 'rev-parse', 'HEAD').stdout.strip()
        actual = git(repo, 'ls-remote', '--heads', 'origin', 'refs/heads/' + branch).stdout.split()
        if not actual or actual[0] != head:
            continue
        return {'ok': True, 'head': head, 'committed': committed, 'pushed': bool(ahead)}
    raise RuntimeError('Remote kept changing; retry on the next cycle')


def run(config):
    lock = (HERE / 'run.lock').open('a+')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        return {'ok': True, 'skipped': 'another run is active'}
    report = {'started_at': now(), 'repo': config['repo']}
    try:
        report.update(synchronize(config))
        if report.get('committed') or report.get('pushed'):
            LOG.info('Synchronized %s', report['head'])
    except Exception as error:
        report.update(ok=False, error=str(error))
        LOG.error('%s', error)
    finally:
        report['finished_at'] = now()
        save(HERE / 'status.json', report)
        lock.close()
    return report


def daemon(config):
    lock = (HERE / 'daemon.lock').open('a+')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        return
    stopping = [False]
    def stop(*_):
        stopping[0] = True
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    save(HERE / 'daemon.json', {'pid': os.getpid(), 'started_at': now(), 'interval_seconds': config['interval_seconds']})
    while not stopping[0]:
        try:
            run(config)
        except Exception:
            LOG.exception('Synchronization cycle failed')
        deadline = time.monotonic() + config['interval_seconds']
        while not stopping[0] and time.monotonic() < deadline:
            time.sleep(min(1, max(0, deadline - time.monotonic())))


def ensure():
    # Lock held by daemon, so duplicate boot/login hooks cannot create two workers.
    guard = (HERE / 'start.lock').open('a+')
    fcntl.flock(guard, fcntl.LOCK_EX)
    probe = (HERE / 'daemon.lock').open('a+')
    try:
        fcntl.flock(probe, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        return
    probe.close()
    with (HERE / 'daemon.stdout.log').open('ab') as output:
        child = subprocess.Popen(['/usr/bin/python3', str(HERE / 'sync.py'), 'daemon'],
                                 stdin=subprocess.DEVNULL, stdout=output, stderr=output,
                                 start_new_session=True, close_fds=True)
    for _ in range(50):
        if child.poll() is not None:
            raise RuntimeError('Daemon did not start')
        try:
            if json.loads((HERE / 'daemon.json').read_text())['pid'] == child.pid:
                return
        except (FileNotFoundError, ValueError, KeyError):
            pass
        time.sleep(0.1)
    raise RuntimeError('Daemon startup timed out')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['run', 'daemon', 'ensure', 'status'])
    action = parser.parse_args().action
    handler = RotatingFileHandler(str(HERE / 'sync.log'), maxBytes=2*1024*1024, backupCount=3, encoding='utf-8')
    handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(message)s'))
    LOG.addHandler(handler)
    LOG.setLevel(logging.INFO)
    config = json.loads((HERE / 'config.json').read_text())
    if action == 'daemon':
        daemon(config)
    elif action == 'ensure':
        ensure()
    elif action == 'status':
        report = json.loads((HERE / 'status.json').read_text())
        report['daemon'] = json.loads((HERE / 'daemon.json').read_text())
        print(json.dumps(report, ensure_ascii=False))
    else:
        report = run(config)
        print(json.dumps(report, ensure_ascii=False))
        if not report['ok']:
            raise SystemExit(1)


if __name__ == '__main__':
    main()
