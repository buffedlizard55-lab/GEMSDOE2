/*
 * generator_ui_harness.js — run the how-to-submit page's generator glue in node, with just enough
 * DOM to prove the wiring works.
 *
 * WHY: geotiff_writer.js is already judged headlessly (scripts/check_site_generator.py) and by
 * rasterio. What that does NOT test is the glue a reader actually clicks: that
 * docs/generate_submission.js finds its mount point, fetches the two payload files by the names the
 * site ships, calls the writer, refuses to hand over a file when a self-check fails, and produces
 * the same bytes the CLI produces. Those are the failure modes that would leave a live button
 * printing nothing, and none of them are visible to a Python test.
 *
 * Usage: node tests/support/generator_ui_harness.js <docs-dir> <mode: tif|zip|broken>
 * Prints one JSON object; exit 0 only when the assertions inside it hold.
 */
'use strict';
const fs = require('fs');
const path = require('path');
const vm = require('vm');

const DOCS = path.resolve(process.argv[2] || 'docs');
const MODE = process.argv[3] || 'tif';
const BLOBS = new Map();
let blobSeq = 0;
async function blobBytes(blob) {
  return new Uint8Array(await blob.arrayBuffer());
}

function makeNode(tag) {
  const node = {
    tag, children: [], className: '', textContent: '', href: '', download: '', type: '',
    disabled: false, listeners: {}, dataset: {}, style: {},
    appendChild(c) { this.children.push(c); return c; },
    addEventListener(ev, fn) { (this.listeners[ev] = this.listeners[ev] || []).push(fn); },
    click() { (this.listeners.click || []).forEach((f) => f({ type: 'click' })); },
    get lastChild() { return this.children[this.children.length - 1] || null; },
  };
  // textContent = '' is the code's only way to clear a container, so it must clear the children too
  Object.defineProperty(node, 'textContent', {
    get() { return this._text || ''; },
    set(v) { this._text = String(v); if (v === '') this.children.length = 0; },
  });
  node.classList = { add: () => {}, remove: () => {} };
  return node;
}

const mount = makeNode('div');
const document = {
  getElementById(id) { return id === 'tif-generator' ? mount : null; },
  createElement: makeNode,
};
const fetchLog = [];
async function fakeFetch(url) {
  const rel = String(url).replace(/^\.?\//, '');
  fetchLog.push(rel);
  const p = path.join(DOCS, rel);
  if (!fs.existsSync(p)) return { ok: false, status: 404 };
  const buf = fs.readFileSync(p);
  return {
    ok: true, status: 200,
    json: async () => JSON.parse(buf.toString('utf8')),
    arrayBuffer: async () => buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength),
  };
}

const sandbox = {
  console, setTimeout, clearTimeout, URL: {
    createObjectURL(b) { const k = 'blob:gen-' + (++blobSeq); BLOBS.set(k, b); return k; },
    revokeURL() {},
  },
  // the REAL Blob/Response, not stubs: geotiff_writer.js streams strips through them for deflate,
  // so faking them would test a runtime the browser does not have
  TextEncoder, TextDecoder, Blob: globalThis.Blob, Response: globalThis.Response,
  ReadableStream: globalThis.ReadableStream, WritableStream: globalThis.WritableStream,
  DataView, Uint8Array, Uint32Array, Float32Array, ArrayBuffer,
  Promise, Math, Date, JSON, isFinite, parseInt, parseFloat, String, Number, Object, Array,
  Error, RegExp, document, fetch: fakeFetch,
  crypto: require('crypto').webcrypto,
  CompressionStream: require('stream/web').CompressionStream,
  DecompressionStream: require('stream/web').DecompressionStream,
};
sandbox.window = sandbox;
sandbox.globalThis = sandbox;
vm.createContext(sandbox);

const out = { assertions: [], ok: true };
function assert(name, cond, detail) {
  out.assertions.push({ name, ok: !!cond, detail: String(detail === undefined ? '' : detail) });
  if (!cond) out.ok = false;
}

(async () => {
  // 1. load the writer, then the glue, exactly as the page's <script> tags do (in that order)
  vm.runInContext(fs.readFileSync(path.join(DOCS, 'geotiff_writer.js'), 'utf8'), sandbox,
                  { filename: 'geotiff_writer.js' });
  assert('writer registers GemsGeoTIFF on the global', typeof sandbox.GemsGeoTIFF === 'object',
    typeof sandbox.GemsGeoTIFF);
  vm.runInContext(fs.readFileSync(path.join(DOCS, 'generate_submission.js'), 'utf8'), sandbox,
                  { filename: 'generate_submission.js' });

  await new Promise((r) => setTimeout(r, 400));   // let the boot preload resolve

  // 2. the panel must have built itself into the mount
  assert('mount was found and populated', mount.children.length > 0, mount.children.length + ' nodes');
  const texts = JSON.stringify(mount, (k, v) => (k === 'parent' ? undefined : v));
  assert('glue fetched the manifest by the name the site ships',
    fetchLog.includes('submission_meta.json'), fetchLog.join(','));
  assert('glue did not invent a second payload path',
    fetchLog.every((f) => ['submission_meta.json', 'submission_field.bin'].includes(f)),
    fetchLog.join(','));
  assert('provenance table names the artifact it reproduces',
    texts.includes('ens12-adopted-floor0.1-w0'), 'artifact path rendered');

  // 3. click "Build submission.tif" (or the zip variant) and wait for the status line
  const btnTif = mount.children.find((c) => c.tag === 'div' && c.className === 'gen-actions');
  assert('two build buttons exist', btnTif && btnTif.children.length === 2,
    btnTif ? btnTif.children.length : 'no actions row');
  const btn = btnTif.children[MODE === 'zip' ? 1 : 0];
  btn.click();
  for (let i = 0; i < 300 && !/Ready —|Failed:/.test(statusText()); i++) {
    await new Promise((r) => setTimeout(r, 100));
  }
  function statusText() {
    const s = mount.children.find((c) => c.className && String(c.className).startsWith('gen-status'));
    return s ? s.children.map((c) => c.textContent).join('') : '';
  }
  const status = statusText();
  if (MODE === 'broken') {
    assert('a tampered manifest is refused, not downloaded', /Failed:/.test(status), status);
    assert('no download was offered on the failure', BLOBS.size === 0, BLOBS.size + ' blobs');
    process.stdout.write(JSON.stringify(out, null, 1) + '\n');
    process.exit(out.ok ? 0 : 1);
  }
  assert('status reports success', /Ready —/.test(status), status);
  assert('field blob was fetched when the build was clicked',
    fetchLog.includes('submission_field.bin'), fetchLog.join(','));

  // 4. the bytes handed to the download must match what the CLI produces for the same mode
  const blob = [...BLOBS.values()].pop();
  assert('a blob was registered for download', !!blob, BLOBS.size);
  const resultRow = mount.children.find((c) => c.className === 'gen-result');
  const dl = resultRow && resultRow.children.find((c) => c.className === 'gen-download');
  const link = dl && dl.children.find((c) => c.tag === 'a');
  // The download name is unique per build: UTC instant + the artifact's sha prefix, so a
  // team's Downloads folder and the platform's Note field can always be told apart.
  const wantName = /^gems-submission-\d{8}T\d{6}Z-[0-9a-f]{8}\.(tif|zip)$/;
  const wantExt = MODE === 'zip' ? '.zip' : '.tif';
  assert('download link names a unique per-build file',
    link && wantName.test(link.download) && link.download.endsWith(wantExt),
    link && link.download);
  assert('download type matches the container',
    link && link.href.startsWith('blob:') &&
    blob.type === (MODE === 'zip' ? 'application/zip' : 'image/tiff'), blob && blob.type);

  // 4b. the identity block: unique name + a Note the team can tell submissions apart by
  const ident = resultRow && resultRow.children.find((c) => c.className === 'gen-identity');
  assert('an identity block (unique name + suggested note) was rendered', !!ident,
    ident ? 'present' : 'missing');
  const identTxt = ident ? JSON.stringify(ident) : '';
  assert('the suggested note is policy · build <sha8> · <UTC stamp>',
    /· build [0-9a-f]{8} · \d{8}T\d{6}Z/.test(identTxt), identTxt.slice(0, 300));
  assert('the identity repeats the download name stem',
    new RegExp(link.download.replace(/\.(tif|zip)$/, '')).test(identTxt),
    identTxt.slice(0, 300));

  const { createHash } = require('crypto');
  const bytes = await blobBytes(blob);
  out.generated = { bytes: bytes.length, sha256: createHash('sha256').update(Buffer.from(bytes)).digest('hex') };

  // 5. the self-check table must have been rendered, and every row must be a pass
  const checks = resultRow ? resultRow.children.filter((c) => c.className === 'gen-checks') : [];
  assert('a self-check table was rendered', checks.length === 1, checks.length);
  const rows = checks.length ? checks[0].children.slice(1) : [];
  assert('17 self-checks are shown', rows.length === 17, rows.length + ' rows');
  assert('every self-check row shows the pass glyph',
    rows.every((r) => JSON.stringify(r).includes('"✓"')), 'one or more rows without a tick');

  process.stdout.write(JSON.stringify(out, null, 1) + '\n');
  process.exit(out.ok ? 0 : 1);
})().catch((e) => {
  process.stdout.write(JSON.stringify({ ok: false, error: String(e && e.stack || e), out }, null, 1) + '\n');
  process.exit(3);
});
