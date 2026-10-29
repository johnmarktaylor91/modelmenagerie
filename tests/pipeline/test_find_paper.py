"""Paper discovery, replacement and bounded bibliographic trickles."""
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from menagerie.pipeline import DataLayout, StoreLayout, read_json, write_json
from menagerie.pipeline import bib_harvest, fetch, find_paper, harvest, publish
from menagerie.pipeline.records import Amendment, Availability, Citation, Identity, Member, Packet, RecordError, Search, Source

AT = '2026-10-08T20:00:00Z'
TITLE = 'Example architecture for images'
REF = 'arXiv:2103.14899'

@pytest.fixture()
def roots(tmp_path, monkeypatch):
    monkeypatch.setenv('MENAGERIE_WORK_ROOT', str(tmp_path / 'work'))
    return DataLayout(tmp_path / 'data'), StoreLayout(tmp_path / 'store')

def packet(family='f-example', **changes):
    base = Packet(family=family, written=AT, by='research', members=(Member(id='m1'),),
                  identity=Identity(canonical='Example'),
                  sources=(Source(sid='s1', role='page', url='https://example.org/model'),),
                  availability=Availability(paper='not_found', code='not_found', license='not_found', description='insufficient'),
                  search=Search(places=('official README',)))
    return replace(base, **changes)

def agreement(monkeypatch):
    sources = (Source(sid='b1', role='bib', api='openalex', blob='a' * 64), Source(sid='b2', role='bib', api='bibtex', blob='b' * 64))
    records = {'openalex': {'title': TITLE, 'id': 'https://openalex.org/W1', 'publication_date': '2021-03-01', 'cited_by_count': 12,
                            'authorships': [{'author': {'display_name': 'A Author'}}]},
               'bibtex': f'@misc{{x,\n title={{{TITLE}}},\n year={{2021}}\n}}'}
    monkeypatch.setattr(bib_harvest, '_gather', lambda *a, **k: (fetch.BibResult(sources, (), records), records, ['openalex']))

def test_saved_identifier_deduplicates_title_search(roots, monkeypatch):
    data, store = roots
    write_json(packet(citation=Citation(title=TITLE)), data.packet('f-example'))
    monkeypatch.setattr(find_paper, '_text', lambda *a: [('s1', 'https://arxiv.org/abs/2103.14899')])
    monkeypatch.setattr(find_paper, '_search_refs', lambda *a: [(REF, None)])
    agreement(monkeypatch)
    assert find_paper.find(['f-example'], data=data, store=store)[0]['status'] == 'found'
    got = read_json(Packet, data.packet('f-example'))
    assert got.citation.arxiv == '2103.14899' and got.citation.openalex == 'W1'
    assert got.citation.bib == ('b1', 'b2') and got.members == packet().members

def test_dry_run_preserves_packet(roots, monkeypatch):
    data, store = roots
    path = data.packet('f-example')
    write_json(packet(citation=Citation(title=TITLE)), path)
    before = path.read_bytes()
    monkeypatch.setattr(find_paper, '_search_refs', lambda *a: [(REF, None)])
    agreement(monkeypatch)
    assert find_paper.find(['f-example'], data=data, store=store, write=False)[0]['status'] == 'found'
    assert path.read_bytes() == before

def test_name_alone_never_becomes_title_search(roots, monkeypatch):
    data, store = roots
    write_json(packet(), data.packet('f-example'))
    monkeypatch.setattr(find_paper, '_search_refs', lambda *a: pytest.fail('guessed title'))
    assert find_paper.find(None, data=data, store=store)[0]['status'] == 'needs_2a'

def test_arxiv_refusal_falls_back_to_exact_openalex_title(roots, monkeypatch):
    data, store = roots
    def refused(*a, **k):
        raise fetch.FetchError('unavailable')
    monkeypatch.setattr(find_paper, '_arxiv_title', refused)
    monkeypatch.setattr(bib_harvest, '_title_search', lambda *a, **k: ((), {'title': TITLE, 'doi': 'https://doi.org/10.1234/example'}))
    assert find_paper._search_refs(TITLE, packet(), store, data) == [('10.1234/example', None)]

def test_disagreement_cannot_fill_identifiers(roots, monkeypatch):
    data, store = roots
    agreement(monkeypatch)
    assert find_paper._agree(packet(), REF, 's1', 'Different title', store, data) is None
    records = {'s2': {'title': TITLE, 'paperId': 'P1'}, 'openalex': {'title': 'wrong', 'id': 'Wwrong', 'cited_by_count': 999}}
    assert bib_harvest._filled(Citation(title=TITLE), records, ['s2'], 'test').openalex is None
    assert 'openalex' not in bib_harvest._agreed_records(records, ['s2'])

def test_titleless_mismatch_stays_held(roots, monkeypatch):
    data, store = roots
    write_json(packet('f-unrelated', identity=Identity(canonical='Unrelated')), data.packet('f-unrelated'))
    monkeypatch.setattr(find_paper, '_text', lambda *a: [('s1', 'arXiv:2103.14899')])
    agreement(monkeypatch)
    assert find_paper.find(['f-unrelated'], data=data, store=store)[0]['status'] == 'held'
    assert bib_harvest.citation_held(read_json(Packet, data.packet('f-unrelated')))

def test_replacement_releases_hold_but_excludes_old_sources(roots, monkeypatch):
    data, store = roots
    old = Source(sid='s2', role='paper', url='https://arxiv.org/abs/2001.00001')
    held = Amendment(at=AT, why='titleless: spot check', evidence=('s2',), changed=('citation',), by=bib_harvest.BY)
    write_json(packet(sources=(*packet().sources, old), citation=Citation(title='Wrong', arxiv='2001.00001'), amendments=(held,)), data.packet('f-example'))
    agreement(monkeypatch)
    find_paper.apply('f-example', {'ref': REF, 'title': TITLE, 'evidence': ['s1'], 'reason': 'official README identifies implementation'}, data=data, store=store)
    got = read_json(Packet, data.packet('f-example'))
    assert not bib_harvest.citation_held(got) and bib_harvest.held_sources(got) == {'s2'}
    assert harvest._refused_paper(got, {'url': old.url})
    assert not harvest._refused_paper(got, {'url': 'https://arxiv.org/abs/2103.14899'})

def test_typed_no_paper_preserves_evidence_and_is_terminal(roots):
    data, store = roots
    old = Source(sid='s2', role='paper', url='https://arxiv.org/abs/2001.00001')
    write_json(packet(sources=(*packet().sources, old)), data.packet('f-example'))
    find_paper.apply('f-example', {'no_paper': 'heterogeneous_members', 'evidence': ['s1'], 'reason': 'different works per member'}, data=data, store=store)
    got = read_json(Packet, data.packet('f-example'))
    assert got.citation is None and got.availability.paper == 'not_found'
    assert find_paper.find(None, data=data, store=store)[0]['status'] == 'no_paper'
    assert find_paper.harvest(None, data=data, store=store) == [] and publish._paper_url(got) is None

def test_decision_requires_identity_evidence(roots):
    data, store = roots
    write_json(packet(), data.packet('f-example'))
    with pytest.raises(RecordError, match='source evidence'):
        find_paper.apply('f-example', {'ref': REF, 'reason': 'guess'}, data=data, store=store)

def test_harvest_limit_counts_eligible_and_retries_unseeded_complete(roots, monkeypatch):
    data, store = roots
    write_json(packet('f-a'), data.packet('f-a'))
    cite = Citation(title=TITLE, arxiv='2103.14899', authors=('A',), first_public='2021', venue='Venue', bib=('s1',))
    for family in ('f-b', 'f-c'):
        write_json(packet(family, citation=cite), data.packet(family))
    calls = []
    def fill(family, **kwargs):
        calls.append((family, kwargs['force']))
        return bib_harvest.Outcome(family, 'filled')
    monkeypatch.setattr(bib_harvest, 'fill', fill)
    assert len(find_paper.harvest(None, data=data, store=store, limit=1)) == 1 and calls == [('f-b', True)]
    assert find_paper.harvest(None, data=data, store=store, limit=0) == []
    with pytest.raises(RecordError):
        find_paper.harvest(None, data=data, store=store, limit=-1)

@pytest.mark.parametrize(('partial', 'precise', 'expected'), [('2021', '2021-03-27', '2021-03-27'), ('2021-03', '2021-03-27', '2021-03-27'), ('2020', '2021-03-27', '2020')])
def test_first_public_partial_dates(partial, precise, expected):
    records = {'s2': {'publicationDate': precise, 'year': int(partial[:4])}, 'openalex': {'publication_date': partial}}
    assert bib_harvest._first_public(records, ['s2', 'openalex']) == expected

def test_export_bibtex_uses_only_citation_bib(roots):
    _, store = roots
    old, current = 'a' * 64, 'b' * 64
    for sha, text in ((old, '@misc{wrong}'), (current, '@misc{correct}')):
        store.blob(sha).parent.mkdir(parents=True, exist_ok=True)
        store.blob(sha).write_text(text)
    sources = (Source(sid='s1', role='bib', api='bibtex', blob=old), Source(sid='s2', role='bib', api='bibtex', blob=current))
    got = packet(sources=sources, citation=Citation(title=TITLE, bib=('s2',)))
    assert publish._bibtex(SimpleNamespace(family='f-example'), got, SimpleNamespace(store=store)) == '@misc{correct}'

def test_classics_correction_is_family_specific():
    url, title = harvest.classic_paper('menagerie.classics.gnot', 'build_gnot', family='f-gino')
    assert url.endswith('2309.00583') and title.startswith('Geometry-Informed')
    assert len(harvest._classic_papers()) == 8

def test_shim_arguments_are_keyless(monkeypatch):
    calls = []
    monkeypatch.setenv('OPENALEX_API_KEY', 'sentinel')
    def run(argv, **kwargs):
        calls.append((argv, kwargs))
        return SimpleNamespace(returncode=0, stdout=b'{"id":"W1"}', stderr=b'private diagnostics')
    monkeypatch.setattr(fetch.subprocess, 'run', run)
    assert fetch._openalex('https://api.openalex.org/works?filter=title.search:A%20B').status == 200
    argv, kwargs = calls[0]
    assert argv[-1] == 'filter=title.search:A B' and 'OPENALEX_API_KEY' not in kwargs['env']
    with pytest.raises(fetch.FetchError, match='credential'):
        fetch._openalex('https://api.openalex.org/works?api_key=sentinel')

def test_paper_batches_and_brief_are_bounded(roots):
    data, _ = roots
    families = [f'f-example{i}' for i in range(16)]
    for family in families:
        write_json(packet(family), data.packet(family))
    batches = find_paper.batches(families, data=data)['batches']
    assert [len(b['families']) for b in batches] == [15, 1]
    text = find_paper.brief(batches[1], data=data, lane_dir=data.root / 'lane')
    assert '3 minutes and 10 tool calls' in text and '$0.00' in text


def test_missing_bibtex_falls_back_to_citation(roots):
    _, store = roots
    source = Source(sid='s1', role='bib', api='bibtex', blob='a' * 64)
    got = packet(sources=(source,), citation=Citation(title=TITLE, bib=('s1',)))
    result = publish._bibtex(SimpleNamespace(family='f-example'), got, SimpleNamespace(store=store))
    assert TITLE in result


def test_shim_refuses_ssh_prefix_and_preserves_upstream_status(monkeypatch):
    monkeypatch.setenv('MENAGERIE_OPENALEX_COMMAND', 'ssh host openalex')
    with pytest.raises(fetch.FetchError, match='local broker'):
        fetch._openalex('https://api.openalex.org/works?filter=title.search:Foo%3Btouch%20x')
    monkeypatch.setenv('MENAGERIE_OPENALEX_COMMAND', 'openalex')
    monkeypatch.setattr(fetch.subprocess, 'run', lambda *a, **k:
                        SimpleNamespace(returncode=1, stdout=b'', stderr=b'openalex: HTTP 404\nprivate diagnostics'))
    got = fetch._openalex('https://api.openalex.org/works/W0')
    assert got.status == 404 and b'private' not in got.body


def test_refused_head_does_not_starve_harvest_tail(roots, monkeypatch):
    data, store = roots
    cite = Citation(title=TITLE, arxiv='2103.14899')
    for family in ('f-a', 'f-b'):
        write_json(packet(family, citation=cite), data.packet(family))
    calls = []
    def refused(family, **kw):
        calls.append(family)
        return bib_harvest.Outcome(family, 'refused', note='API unavailable')
    monkeypatch.setattr(bib_harvest, 'fill', refused)
    find_paper.harvest(None, data=data, store=store, limit=1)
    find_paper.harvest(None, data=data, store=store, limit=1)
    assert calls == ['f-a', 'f-b']
    assert find_paper.harvest(None, data=data, store=store, limit=1) == []


def test_apply_supersedes_no_paper_and_rejects_path_traversal(roots, monkeypatch):
    data, store = roots
    write_json(packet(), data.packet('f-example'))
    find_paper.apply('f-example', {'no_paper':'unpublished', 'evidence':['s1'], 'reason':'searched'}, data=data, store=store)
    agreement(monkeypatch)
    find_paper.apply('f-example', {'ref':REF, 'title':TITLE, 'evidence':['s1'], 'reason':'official README now names the paper'}, data=data, store=store)
    got = read_json(Packet, data.packet('f-example'))
    assert not bib_harvest.no_paper(got) and got.availability.paper == 'found'
    with pytest.raises(RecordError, match='canonical family'):
        find_paper.find(['../../x'], data=data, store=store)


def test_harvester_does_not_restore_removed_citation_or_refused_link(roots, monkeypatch):
    data, _ = roots
    monkeypatch.setattr(harvest, 'june_notes', lambda *a: '')
    amendment = Amendment(at=AT, why='no_paper:heterogeneous_members; examined members', evidence=('s1',), changed=('citation',), by=find_paper.BY)
    old = packet(by='harvester@x', amendments=(amendment,))
    write_json(old, data.packet(old.family))
    row = harvest.Row('m1', 'Example', 'classics', '', 'Example')
    sources = [{'role':'code','url':'menagerie/classics/example.py'},
               {'role':'paper','url':'https://arxiv.org/abs/2001.00001','_title':'Wrong paper'}]
    got, _ = harvest._packet(old.family, [harvest.Draft(row, 'classic', {}, sources)], data)
    assert got.citation is None and got.availability.paper == 'not_found'
    assert not any(s.role == 'paper' for s in got.sources)


def test_parsed_title_mismatch_requires_identity_check(roots, monkeypatch):
    data, store = roots
    write_json(packet('f-unrelated', identity=Identity(canonical='Unrelated')), data.packet('f-unrelated'))
    monkeypatch.setattr(find_paper, '_text', lambda *a: [('s1', f'@misc{{x, title={{{TITLE}}}, year={{2021}}}} arXiv:2103.14899')])
    monkeypatch.setattr(find_paper, '_search_refs', lambda *a: [])
    agreement(monkeypatch)
    assert find_paper.find(['f-unrelated'], data=data, store=store)[0]['status'] == 'held'


def test_decision_rejects_retired_evidence_and_malformed_sources(roots):
    data, store = roots
    retired = Amendment(at=AT, why='wrong paper', evidence=('s1',), changed=('sources',), by=bib_harvest.RETIRED)
    write_json(packet(amendments=(retired,)), data.packet('f-example'))
    decision = {'no_paper': 'unpublished', 'evidence': ['s1'], 'reason': 'searched'}
    with pytest.raises(RecordError, match='held or retired'):
        find_paper.apply('f-example', decision, data=data, store=store)
    with pytest.raises(RecordError, match='list of source objects'):
        find_paper.apply('f-example', {**decision, 'sources': {}}, data=data, store=store)


@pytest.mark.parametrize('status,body,note', [(404, b'{}', 'not_found'), (429, b'{"error":"insufficient budget"}', 'refused (insufficient budget)')])
def test_openalex_negative_lookup_is_cached_for_day(roots, monkeypatch, status, body, note):
    calls = []
    monkeypatch.setattr(fetch, '_event', lambda *a, **k: None)
    def response(*a, **k):
        calls.append(a)
        return fetch.Response(status, 'https://api.openalex.org/works/W0', 'application/json', body)
    monkeypatch.setattr(fetch, '_http', response)
    for _ in range(2):
        got, reason = fetch._bib_get('openalex', 'https://api.openalex.org/works/W0', family='f-example')
        assert got is None and reason == f'openalex:{note}'
    assert len(calls) == 1


def test_title_search_cannot_attach_an_unrelated_candidate_identifier(roots, monkeypatch):
    data, store = roots
    records = {'bibtex': '@misc{x,\n title={Unrelated paper},\n}\n',
               'openalex': {'title': 'Unrelated paper', 'id': 'Wwrong'},
               'openalex_search': {'title': TITLE, 'id': 'Wcorrect'}}
    got = fetch.BibResult((), (), records)
    monkeypatch.setattr(bib_harvest, '_gather', lambda *a, **k: (got, records, ['openalex']))
    assert find_paper._agree(packet(citation=Citation(title=TITLE)), REF, 's1', TITLE, store, data) is None


def test_malformed_decision_is_a_cli_usage_error(tmp_path, capsys):
    from menagerie.pipeline import cli
    path = tmp_path / 'bad.json'
    path.write_text('{bad')
    with pytest.raises(SystemExit) as raised:
        cli.main(['paper', 'apply', 'f-example', str(path)])
    assert raised.value.code == 2
    error = capsys.readouterr().err
    assert 'invalid paper decision' in error and 'Traceback' not in error


def test_held_citation_does_not_readmit_its_unverified_identifier():
    url = 'https://arxiv.org/abs/2103.14899'
    hold = Amendment(at=AT, why='titleless: spot check', evidence=('s2',), changed=('citation',), by=bib_harvest.BY)
    old = packet(citation=Citation(title=TITLE, arxiv='2103.14899'),
                 sources=(*packet().sources, Source(sid='s2', role='paper', url=url)), amendments=(hold,))
    assert harvest._refused_paper(old, {'url': url})


def _unmatched(calls):
    def fill(family, **kw):
        calls.append(family)
        return bib_harvest.Outcome(family, 'unmatched', note='agreeing records give no missing field')
    return fill


def test_harvest_waits_after_nothing_new_unless_inputs_change_and_never_for_good(roots, monkeypatch):
    data, store = roots
    write_json(packet('f-a', citation=Citation(title=TITLE, arxiv='2103.14899')), data.packet('f-a'))
    calls, clock = [], ['2026-10-10T00:00:00Z']
    monkeypatch.setattr(bib_harvest, 'fill', _unmatched(calls))
    monkeypatch.setattr(find_paper, 'now', lambda: clock[0])
    find_paper.harvest(None, data=data, store=store)
    marker = read_json(Packet, data.packet('f-a')).amendments[-1]
    assert calls == ['f-a'] and find_paper.INPUTS.search(marker.why)
    clock[0] = '2026-10-11T00:00:00Z'  # the next UTC day, nothing changed: no second round
    assert find_paper.harvest(None, data=data, store=store) == [] and calls == ['f-a']
    got = read_json(Packet, data.packet('f-a'))  # an input changes: due again at once
    write_json(replace(got, citation=replace(got.citation, doi='10.1234/example')), data.packet('f-a'))
    assert len(find_paper.harvest(None, data=data, store=store)) == 1 and calls == ['f-a', 'f-a']
    assert find_paper.harvest(None, data=data, store=store) == []
    wait = find_paper._retry_days('f-a', marker.why)
    assert find_paper.RETRY_DAYS <= wait < find_paper.RETRY_DAYS + find_paper.RETRY_SPREAD_DAYS
    clock[0] = (datetime(2026, 10, 11, tzinfo=timezone.utc) + timedelta(days=wait)).strftime('%Y-%m-%dT%H:%M:%SZ')
    assert len(find_paper.harvest(None, data=data, store=store)) == 1  # never dropped for good


def test_unkeyed_markers_and_passing_refusals_wait_their_own_time(roots, monkeypatch):
    data, store = roots
    cite = Citation(title=TITLE, arxiv='2103.14899')
    legacy = Amendment(at='2026-10-10T00:00:00Z', why='paper harvest: unmatched; agreeing records give no missing field',
                       evidence=(), changed=('paper_harvest',), by=find_paper.HARVEST_BY)
    later = Amendment(at='2026-10-10T05:00:00Z', why='drafts merged', evidence=('s1',), changed=('written',), by='research')
    write_json(packet('f-a', citation=cite, amendments=(legacy,)), data.packet('f-a'))
    write_json(packet('f-b', citation=cite, amendments=(legacy, later)), data.packet('f-b'))
    calls, clock = [], ['2026-10-11T00:00:00Z']
    def refused(family, **kw):
        calls.append(family)
        return bib_harvest.Outcome(family, 'refused', note='openalex:http_503')
    monkeypatch.setattr(bib_harvest, 'fill', refused)
    monkeypatch.setattr(find_paper, 'now', lambda: clock[0])
    find_paper.harvest(None, data=data, store=store)
    assert calls == ['f-b']  # f-a is unchanged since its marker; f-b changed after it
    clock[0] = '2026-10-12T00:00:00Z'
    assert find_paper.harvest(None, data=data, store=store) == []
    clock[0] = '2026-10-13T00:00:01Z'  # a passing refusal is retried after two days
    find_paper.harvest(None, data=data, store=store)
    assert calls == ['f-b', 'f-b']
    lasting = 'paper harvest: refused; inputs 0123456789abcdef; shared docstring: the module names several papers'
    assert find_paper._retry_days('f-b', lasting) >= find_paper.RETRY_DAYS


def test_a_harvest_without_write_lists_due_packets_and_writes_nothing(roots, monkeypatch):
    data, store = roots
    write_json(packet('f-a', citation=Citation(title=TITLE, arxiv='2103.14899')), data.packet('f-a'))
    before = data.packet('f-a').read_bytes()
    def boom(*a, **k):
        raise AssertionError('a dry run must not fetch')
    monkeypatch.setattr(bib_harvest, 'fill', boom)
    monkeypatch.setattr(fetch, 'bib_records', boom)
    assert find_paper.harvest(None, data=data, store=store, write=False) == [
        {'family': 'f-a', 'status': 'due', 'ref': REF}]
    assert data.packet('f-a').read_bytes() == before and not store.root.exists()
    assert sorted(p.name for p in data.root.rglob('*') if p.is_file()) == ['f-a.json']
