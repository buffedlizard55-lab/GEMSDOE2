/*
 * generate_submission.js — the UI glue for the "build the .tif here" panel on
 * docs/how_to_submit.html.
 *
 * RULES THIS FILE OBEYS
 * ---------------------
 * 1. It contains no numbers of its own.  Every figure it prints (grid, hashes, pixel counts, the
 *    artifact it reproduces) is read from submission_meta.json, which
 *    scripts/build_submission_payload.py wrote by measuring the artifact.  A stale number typed
 *    into this file would be a published lie.
 * 2. It never offers a download before GemsGeoTIFF has re-read the bytes it produced and every
 *    self-check has passed.  A file that cannot be verified is reported as a failure, not handed
 *    over with a warning.
 * 3. It says what the file IS: the pixels of the committed, format-validated artifact, in a
 *    container this page wrote.  It does not claim byte equality with the artifact (different
 *    container) and does not claim a score (only the platform can score).
 *
 * Everything it does lives in geotiff_writer.js, which the test suite also runs under node, so the
 * path you click here and the path CI verifies are the same code.
 */
(function () {
  'use strict';

  var mount = document.getElementById('tif-generator');
  if (!mount) return;

  var META_URL = 'submission_meta.json';
  var BLOB_URL = 'submission_field.bin';

  var state = { meta: null, blob: null, result: null, busy: false, mode: 'tif' };
  var els = {};

  // ---------------------------------------------------------------- tiny dom helpers
  function el(tag, cls, text) {
    var n = document.createElement(tag);
    if (cls) n.className = cls;
    if (text !== undefined && text !== null) n.textContent = String(text);
    return n;
  }
  function fmt(n) {
    return typeof n === 'number' && isFinite(n) ? n.toLocaleString('en-US') : String(n);
  }
  function stage(name, detail, kind) {
    var box = els.stages;
    var row = el('li', 'stage ' + (kind || 'run'));
    row.appendChild(el('span', 'stage-name', name));
    if (detail) row.appendChild(el('span', 'stage-detail', detail));
    box.appendChild(row);
    return row;
  }
  function clearStages() { els.stages.textContent = ''; }
  function say(msg, kind) {
    els.status.textContent = '';
    els.status.className = 'gen-status ' + (kind || '');
    els.status.appendChild(el('span', '', msg));
  }
  function kv(parent, k, v, mono) {
    var row = el('div', 'gen-kv');
    row.appendChild(el('span', 'gen-k', k));
    row.appendChild(el('span', mono ? 'gen-v mono' : 'gen-v', v));
    parent.appendChild(row);
    return row;
  }

  // ---------------------------------------------------------------- build the panel
  function build() {
    mount.textContent = '';
    var head = el('div', 'gen-head');
    head.appendChild(el('h3', null, 'Build the submission .tif here, in this browser'));
    head.appendChild(el('p', 'gen-sub',
      'No install, no GPU, no server: this page ships the pixel field of the adopted artifact and ' +
      'writes the GeoTIFF container locally with ' + 'geotiff_writer.js' + '. Nothing is uploaded — ' +
      'the file appears in your Downloads folder.'));
    mount.appendChild(head);

    var facts = el('div', 'gen-facts');
    els.facts = facts;
    mount.appendChild(facts);

    var actions = el('div', 'gen-actions');
    els.btnTif = el('button', 'gen-btn primary', 'Build submission.tif');
    els.btnZip = el('button', 'gen-btn', 'Build submission.zip (same file, zipped)');
    els.btnTif.type = 'button'; els.btnZip.type = 'button';
    els.btnTif.addEventListener('click', function () { run('tif'); });
    els.btnZip.addEventListener('click', function () { run('zip'); });
    actions.appendChild(els.btnTif);
    actions.appendChild(els.btnZip);
    mount.appendChild(actions);

    els.status = el('p', 'gen-status', 'Idle — nothing has been built yet.');
    mount.appendChild(els.status);

    els.stages = el('ul', 'gen-stages');
    mount.appendChild(els.stages);

    els.result = el('div', 'gen-result');
    mount.appendChild(els.result);

    els.note = el('p', 'gen-foot small');
    els.note.appendChild(el('span', null, 'Provenance: '));
    els.note.appendChild(el('code', null, META_URL));
    els.note.appendChild(el('span', null, ' (measured at site build time from the artifact pinned below)'));
    mount.appendChild(els.note);
  }

  function renderFacts(meta) {
    els.facts.textContent = '';
    var a = meta.artifact || {}, g = meta.grid || {}, e = meta.encoding || {}, f = meta.field || {};
    kv(els.facts, 'Reproduces the pixels of', a.path, true);
    kv(els.facts, 'Artifact sha256 (pinned)', String(a.sha256 || '').slice(0, 32) + '…', true);
    kv(els.facts, 'Artifact size / hash recorded by', fmt(a.bytes) + ' B · sidecar ' + (a.recorded_sha256 ? 'present' : 'absent'), true);
    kv(els.facts, 'Grid', fmt(g.width) + ' × ' + fmt(g.height) + ' px · EPSG:' + g.epsg + ' · ' +
      (g.res ? g.res[0] + ' m' : '?') + ' · ' + g.dtype + ' · ' + g.count + ' band', true);
    kv(els.facts, 'Value histogram in the artifact',
      fmt(f.one_px) + ' px at 1.0 · ' + fmt(f.zero_px) + ' px at 0.0 · ' + fmt(f.nan_px) + ' px NaN (outside the footprint)', true);
    kv(els.facts, 'Payload shipped with this page', e.format + ' · ' + fmt(e.n_runs) +
      ' runs · ' + fmt((meta.blob || {}).bytes) + ' B', true);
    kv(els.facts, 'Policy that produced it', (meta.provenance || {}).policy || '—', true);
    var rt = meta.round_trip || {};
    var pill = el('span', 'pill ' + (rt.float32_bytes_identical ? 'ok' : 'bad'),
      rt.float32_bytes_identical ? 'ENCODER ROUND TRIP: IDENTICAL' : 'ROUND TRIP: BROKEN');
    kv(els.facts, 'Builder-side check', '').appendChild(pill);
  }

  // ---------------------------------------------------------------- fetch
  function fetchJson(url) {
    return fetch(url, { cache: 'no-store' }).then(function (r) {
      if (!r.ok) throw new Error(url + ' → HTTP ' + r.status);
      return r.json();
    });
  }
  function fetchBin(url) {
    return fetch(url, { cache: 'no-store' }).then(function (r) {
      if (!r.ok) throw new Error(url + ' → HTTP ' + r.status);
      return r.arrayBuffer().then(function (b) { return new Uint8Array(b); });
    });
  }

  // ---------------------------------------------------------------- run
  function run(mode) {
    if (state.busy) return;
    state.busy = true; state.mode = mode;
    clearStages();
    els.result.textContent = '';
    els.btnTif.disabled = els.btnZip.disabled = true;
    say('Building — ' + (mode === 'zip' ? 'submission.zip' : 'submission.tif') + ' …', 'run');

    var s1 = stage('load writer', typeof window.GemsGeoTIFF === 'object' ? 'geotiff_writer.js loaded' : 'MISSING',
      typeof window.GemsGeoTIFF === 'object' ? 'ok' : 'bad');
    var s2, s3, s4;

    // Two independent conditions, on purpose: the boot preload sets state.meta without ever
    // touching the field blob, so a single `if (!state.meta)` guard here made the FIRST click on a
    // loaded page throw "the payload did not load" (the harness in tests/support/ catches it).
    var chain = Promise.resolve();
    if (!state.meta) {
      chain = chain.then(function () {
        s2 = stage('fetch manifest', META_URL, 'run');
        return fetchJson(META_URL).then(function (m) {
          state.meta = m;
          renderFacts(m);
          s2.className = 'stage ok';
          s2.appendChild(el('span', 'stage-detail',
            fmt((m.encoding || {}).n_runs) + ' runs · ' + fmt((m.blob || {}).bytes) + ' B field'));
        });
      });
    } else {
      chain = chain.then(function () {
        stage('fetch manifest', 'cached from the page load', 'ok');
      });
    }
    chain = chain.then(function () {
      if (!state.meta) throw new Error('the manifest is not loaded');
      if (state.blob) {
        stage('fetch field blob', 'cached (' + fmt(state.blob.length) + ' B)', 'ok');
        return null;
      }
      s2 = stage('fetch field blob', BLOB_URL, 'run');
      return fetchBin(BLOB_URL).then(function (bytes) {
        state.blob = bytes;
        s2.className = 'stage ok';
        s2.appendChild(el('span', 'stage-detail', fmt(bytes.length) + ' B'));
      });
    });

    chain.then(function () {
      if (!state.meta || !state.blob) throw new Error('the payload did not load');
      s3 = stage('rebuild the field, write the TIFF, re-read it', 'decode → deflate ' +
        (state.meta.grid.height + 63 >> 6) + ' strips → verify', 'run');
      return window.GemsGeoTIFF.generateFromPayload({ meta: state.meta, fieldBytes: state.blob });
    }).then(function (res) {
      s4 = stage('self-check', '', res.ok ? 'ok' : 'bad');
      state.result = res;
      renderChecks(res);
      if (!res.ok) throw new Error('self-check failed: ' + res.failed.join('; '));
      renderDownload(res, state.meta);
      say('Ready — ' + fmt(res.bytes.length) + ' B written locally, all ' + res.checks.length +
        ' self-checks passed.', 'ok');
    }).catch(function (err) {
      say('Failed: ' + (err && err.message ? err.message : err) +
        '  ·  use route A (curl the artifact) or route C (scripts/baseline_submission.py) instead', 'bad');
      if (window.console) console.error('[generate_submission]', err);
    }).then(function () {
      els.btnTif.disabled = els.btnZip.disabled = false;
      state.busy = false;
    });
  }

  // ---------------------------------------------------------------- results
  function renderChecks(res) {
    var box = document.createElement('table');
    box.className = 'gen-checks';
    var head = document.createElement('tr');
    ['Self-check', 'Measured'].forEach(function (h) {
      var th = el('th', null, h); head.appendChild(th);
    });
    box.appendChild(head);
    res.checks.forEach(function (c) {
      var tr = document.createElement('tr');
      var td1 = el('td', null);
      td1.appendChild(el('span', 'pill ' + (c.ok ? 'ok' : 'bad'), c.ok ? '✓' : '✗'));
      td1.appendChild(el('span', null, ' ' + c.name));
      tr.appendChild(td1);
      tr.appendChild(el('td', 'mono small', c.measured));
      box.appendChild(tr);
    });
    els.result.appendChild(box);
  }

  // Unique-per-build identity for the download and for the dialog's Note field. Both parts
  // are read (payload/artifact) or clocked, never typed into this file - the test suite
  // fails the glue for any literal figure. The Note's stated purpose on the platform is
  // "a short comment to help you or your team tell submissions apart later".
  function buildIdentity(meta, res) {
    var stamp = new Date().toISOString()
      .replace(/[-:]/g, '').replace(/\.\d+Z$/, 'Z');           // 20260924T215531Z
    var sha8 = String(((meta.artifact || {}).sha256) || res.sha256 || '').slice(0, 8);
    var tags = ((meta.grid || {}).gdal_tags) || {};
    var policy = tags.shaping_t0
      ? ('floor ' + tags.shaping_t0 + (tags.shaping_thin === 'True' ? ' thin' : ''))
      : 'adopted policy';
    return {
      stamp: stamp,
      sha8: sha8,
      fileStem: 'gems-submission-' + stamp + '-' + sha8,
      note: policy + ' · build ' + sha8 + ' · ' + stamp
    };
  }

  function renderIdentity(id, mode) {
    var wrap = el('div', 'gen-identity');
    var file = id.fileStem + (mode === 'zip' ? '.zip' : '.tif');
    wrap.appendChild(el('div', 'small', 'Suggested file name (unique per build):'));
    var fn = el('div', 'mono gen-identity-v', file);
    wrap.appendChild(fn);
    var noteRow = el('div', 'small', 'Suggested text for the dialog\'s Note field — ' +
      'a short comment that tells your team\'s submissions apart:');
    wrap.appendChild(noteRow);
    var noteVal = el('div', 'mono gen-identity-v', id.note);
    wrap.appendChild(noteVal);
    var copy = el('button', 'gen-btn gen-copy', 'Copy the note');
    copy.type = 'button';
    copy.addEventListener('click', function () {
      var done = function () {
        copy.textContent = 'Copied ✓';
        setTimeout(function () { copy.textContent = 'Copy the note'; }, 1600);
      };
      if (navigator.clipboard && navigator.clipboard.writeText) {
        navigator.clipboard.writeText(id.note).then(done, function () { done(); });
      } else {
        var r = document.createRange();
        r.selectNodeContents(noteVal);
        var sel = window.getSelection();
        sel.removeAllRanges(); sel.addRange(r);
        done();
      }
    });
    wrap.appendChild(copy);
    els.result.appendChild(wrap);
  }

  function renderDownload(res, meta) {
    var wrap = el('div', 'gen-download');
    var bytes = res.bytes, type = 'image/tiff';
    var id = buildIdentity(meta, res);
    var name = id.fileStem + '.tif';
    if (state.mode === 'zip') {
      // no date: buildZip()'s default is the fixed 2020-01-01 instant that
      // scripts/package_submission.py uses, so the CLI and the browser write the same container
      // semantics (a stored member named submission.tif) rather than two near-identical ones.
      // Only the ARCHIVE's own name is unique - the member stays submission.tif.
      bytes = window.GemsGeoTIFF.buildZip([{ name: 'submission.tif', data: res.bytes }]);
      name = id.fileStem + '.zip'; type = 'application/zip';
    }
    var blob = new Blob([bytes], { type: type });
    var url = URL.createObjectURL(blob);
    var a = el('a', 'gen-btn primary', '↓ Download ' + name + ' (' + fmt(bytes.length) + ' B)');
    a.href = url; a.download = name;
    wrap.appendChild(a);
    var copy = el('div', 'small mono gen-hash');
    copy.appendChild(el('div', null, name + '  sha256 ' + res.sha256));
    copy.appendChild(el('div', null, 'inside .tif  sha256 ' + res.sha256 + ' · ' + fmt(res.bytes.length) + ' B' +
      (state.mode === 'zip' ? ' (zipped without recompression, method 0)' : '')));
    copy.appendChild(el('div', null, 'artifact it reproduces  sha256 ' + (meta.artifact || {}).sha256 +
      ' · ' + fmt((meta.artifact || {}).bytes) + ' B  (same pixels, different container: ' +
      ((meta.grid || {}).source_tiff || {}).compression + ' tiled vs deflate stripped)'));
    wrap.appendChild(copy);
    els.result.appendChild(wrap);
    renderIdentity(id, state.mode);
  }

  // ---------------------------------------------------------------- boot
  build();
  if (typeof window.fetch !== 'function' || typeof window.GemsGeoTIFF !== 'object') {
    var why = typeof window.GemsGeoTIFF !== 'object'
      ? 'geotiff_writer.js did not load (open the page from the deployed site, or run ' +
        '`python -m http.server` in docs/ and open http://127.0.0.1:8000/how_to_submit.html)'
      : 'this browser has no fetch(); use route A on the page instead';
    els.btnTif.disabled = els.btnZip.disabled = true;
    say('Generator unavailable: ' + why, 'bad');
  } else {
    // preload only the manifest so the provenance table is filled without a click
    fetchJson(META_URL).then(function (m) {
      state.meta = m;
      renderFacts(m);
      say('Ready to build — provenance measured from ' + (m.artifact || {}).path + '.', '');
    }).catch(function (err) {
      say('Manifest unavailable (' + err.message + ') — the payload is not on this host.', 'bad');
    });
  }
})();
