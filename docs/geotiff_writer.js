/*
 * gems-geotiff-writer.js — build a valid single-band float32 GeoTIFF from the shipped payload.
 *
 * WHY THIS FILE EXISTS
 * --------------------
 * The competition wants one file: a single-band float32 GeoTIFF on the training grid
 * (EPSG:32611, 100 m, 3292 x 3730, NaN outside the GeoDAWN footprint).  Every route in
 * docs/how_to_submit.html to obtain it needs a Python environment.  This file makes one route
 * need nothing at all: the browser decodes docs/submission_field.bin, writes the TIFF container,
 * and re-reads its own bytes to prove the result before it is offered as a download.
 *
 * The test suite runs THIS FILE under node (tests/test_site_generator.py calls it as a CLI),
 * so "verified in CI" and "what your browser does" are literally the same JavaScript, not a
 * Python re-implementation of it.
 *
 * WHAT IT WRITES
 * --------------
 * Classic little-endian TIFF: header, one IFD, out-of-line value blocks, then strips.  Strips are
 * raw-deflated through CompressionStream when the runtime has one (compression tag 32946); if it
 * does not, they are written uncompressed (tag 1).  Both are valid; the reader handles both.
 * Geo-referencing is GeoTIFF 1.0 style: ModelPixelScale(33550) + ModelTiePoint(33922) +
 * GeoKeyDirectory(34735) with ProjectedCSTypeGeoKey = 32611, plus GDAL_NODATA(42113) = "nan" and
 * GDAL_METADATA(42112) carrying AREA_OR_POINT=Area - the tags the artifact itself carries.
 *
 * IT DOES NOT CLAIM TO REPRODUCE THE ARTIFACT'S BYTES.  It reproduces the *pixels* (verified
 * bit-for-bit against the pinned float32 buffer) in a different, independently valid container.
 */
(function (global) {
  'use strict';

  var NAN_U32 = 0x7FC00000;                        // quiet NaN as stored in a little-endian float32
  var T_BYTE = 1, T_ASCII = 2, T_SHORT = 3, T_LONG = 4, T_DOUBLE = 12;
  var SIZEOF = {}; SIZEOF[T_BYTE] = 1; SIZEOF[T_ASCII] = 1; SIZEOF[T_SHORT] = 2;
  SIZEOF[T_LONG] = 4; SIZEOF[T_DOUBLE] = 8;
  /*
   * Compression tag: 8 (AdobeDeflate).  MEASURED, not assumed: `rasterio ... compress=deflate`
   * writes tag 8 with a zlib-wrapped stream (first two strip bytes 78 9c), while a *raw* deflate
   * stream in the same file is rejected by libtiff ("ZIPDecode:Decoding error at scanline 0").
   * 32946 is accepted on read for tolerance, but only 8 is written, so what the page produces is
   * what GDAL itself produces.
   */
  var COMPRESSION_NONE = 1, COMPRESSION_DEFLATE = 8, COMPRESSION_DEFLATE_ALT = 32946;

  // -------------------------------------------------------------- uleb128 run decoder
  function decodeRuns(bytes, width, height) {
    var target = width * height;
    var out = new Float32Array(target);
    var u32 = new Uint32Array(out.buffer);         // NaN is written by bits: `v === NaN` is never true
    var pos = 0, n = 0, runs = 0;
    while (pos < bytes.length) {
      var v = 0, shift = 0, b;
      do {
        if (pos >= bytes.length) throw new Error('truncated uleb128 at run ' + runs);
        b = bytes[pos++];
        v += (b & 0x7f) * Math.pow(2, shift);       // multiplication, not <<: lengths pass 2^31
        shift += 7;
        if (shift > 49) throw new Error('unreasonable run length at run ' + runs);
      } while (b & 0x80);
      if (pos >= bytes.length) throw new Error('run ' + runs + ' has no value code');
      var code = bytes[pos++];
      if (v === 0) throw new Error('zero-length run at ' + runs);
      if (n + v > target) throw new Error('run ' + runs + ' (length ' + v + ') overruns the grid');
      if (code === 2) { for (var i = 0; i < v; i++) u32[n + i] = NAN_U32; }
      else if (code === 0) out.fill(0, n, n + v);
      else if (code === 1) out.fill(1, n, n + v);
      else throw new Error('unknown value code ' + code + ' at run ' + runs);
      n += v; runs++;
      if (n === target && pos < bytes.length) {
        throw new Error((bytes.length - pos) + ' trailing bytes after the grid was filled');
      }
    }
    if (n !== target) throw new Error('runs cover ' + n + ' px of a ' + target + ' px grid');
    return { field: out, runs: runs };
  }

  // -------------------------------------------------------------- raw deflate / inflate
  function hasStreams() {
    return typeof global.CompressionStream === 'function' &&
           typeof global.DecompressionStream === 'function' &&
           typeof global.Blob === 'function' && typeof global.Response === 'function';
  }
  async function deflateZlib(u8) {
    if (!hasStreams()) return null;
    try {
      // 'deflate' in the Web Compression Streams API is the zlib format (RFC 1950): the two-byte
      // header and adler32 trailer that libtiff's Zip codec expects inside a TIFF strip.
      var stream = new global.Blob([u8]).stream().pipeThrough(new global.CompressionStream('deflate'));
      var buf = await new global.Response(stream).arrayBuffer();
      var out = new Uint8Array(buf);
      return out.length < u8.length ? out : null;  // never "compress" something bigger
    } catch (err) { return null; }
  }
  async function inflateZlib(u8, wrapped) {
    if (typeof global.DecompressionStream !== 'function') throw new Error('no DecompressionStream in this runtime');
    var fmt = wrapped === false ? 'deflate-raw' : 'deflate';
    var stream = new global.Blob([u8]).stream().pipeThrough(new global.DecompressionStream(fmt));
    return new Uint8Array(await new global.Response(stream).arrayBuffer());
  }

  // -------------------------------------------------------------- byte helpers
  function enc(str) { return new TextEncoder().encode(str); }
  function zeros(n) { return new Uint8Array(n); }
  function u16(v) { var b = zeros(2); new DataView(b.buffer).setUint16(0, v & 0xffff, true); return b; }
  function u32(v) { var b = zeros(4); new DataView(b.buffer).setUint32(0, v >>> 0, true); return b; }
  function shorts(vals) {
    var b = zeros(vals.length * 2), dv = new DataView(b.buffer);
    for (var i = 0; i < vals.length; i++) dv.setUint16(i * 2, vals[i] & 0xffff, true);
    return b;
  }
  function longs(vals) {
    var b = zeros(vals.length * 4), dv = new DataView(b.buffer);
    for (var i = 0; i < vals.length; i++) dv.setUint32(i * 4, vals[i] >>> 0, true);
    return b;
  }
  function doubles(vals) {
    var b = zeros(vals.length * 8), dv = new DataView(b.buffer);
    for (var i = 0; i < vals.length; i++) dv.setFloat64(i * 8, vals[i], true);
    return b;
  }
  function asciiBytes(str) { var b = zeros(str.length + 1); for (var i = 0; i < str.length; i++) b[i] = str.charCodeAt(i) & 0x7f; return b; }
  function cat(list) {
    var n = 0, i;
    for (i = 0; i < list.length; i++) n += list[i].length;
    var o = zeros(n), p = 0;
    for (i = 0; i < list.length; i++) { o.set(list[i], p); p += list[i].length; }
    return o;
  }
  function align4(n) { return (n + 3) & ~3; }

  /**
   * field  Float32Array (row-major, height*width)
   * grid   the `grid` object from docs/submission_meta.json
   * opts   {compression:'deflate'|'none', rowsPerStrip:int}
   * resolves to {bytes, layout}
   */
  async function buildGeoTIFF(field, grid, opts) {
    opts = opts || {};
    var width = grid.width | 0, height = grid.height | 0;
    if (field.length !== width * height) throw new Error('field is ' + field.length + ' px, grid says ' + (width * height));
    var rowsPerStrip = opts.rowsPerStrip && opts.rowsPerStrip > 0 ? Math.min(opts.rowsPerStrip, height) : 64;
    if (rowsPerStrip > 0xffff) throw new Error('RowsPerStrip must fit a SHORT');
    var nStrips = Math.ceil(height / rowsPerStrip);
    var rawPerStrip = width * rowsPerStrip * 4;
    var pixels = new Uint8Array(field.buffer, field.byteOffset, field.byteLength);

    // ---- split into strips, deflate each one if the runtime lets us -------------------------
    var wantDeflate = opts.compression !== 'none';
    var strips = [], compression = COMPRESSION_NONE, compressedAny = false;
    for (var s = 0; s < nStrips; s++) {
      var from = s * rawPerStrip, to = Math.min(pixels.length, from + rawPerStrip);
      strips.push({ raw: pixels.subarray(from, to), bytes: pixels.subarray(from, to) });
    }
    if (wantDeflate) {
      var all = await Promise.all(strips.map(function (st) { return deflateZlib(st.raw); }));
      if (all.every(function (c) { return !!c; })) {
        for (var k = 0; k < all.length; k++) strips[k].bytes = all[k];
        compression = COMPRESSION_DEFLATE;
        compressedAny = true;
      }
    }

    // ---- tag values ------------------------------------------------------------------------
    var res = grid.res || [100, 100];
    var pixelScale = grid.pixelscale || [res[0], Math.abs(res[1]), 0];
    var tiePoint = grid.tiepoint || [0, 0, 0, grid.origin[0], grid.origin[1], 0];
    var geoKeys = [1, 1, 0, 3,                    // KeyDirectoryVersion, Revision, Minor, NumberOfKeys
                   1024, 0, 1, 1,                 // GTModelTypeGeoKey  = 1 (Projected)
                   1025, 0, 1, 1,                 // GTRasterTypeGeoKey = 1 (AreaPixel)
                   3072, 0, 1, grid.epsg | 0];    // ProjectedCSTypeGeoKey = EPSG
    var nodataBytes = asciiBytes('nan');
    var xmlBytes = asciiBytes('<GDALMetadata>\n  <Item name="AREA_OR_POINT">Area</Item>\n</GDALMetadata>\n');
    var descBytes = grid.band_description ? asciiBytes(grid.band_description) : null;

    var specs = [
      { tag: 256, type: T_LONG, value: width },                        // ImageWidth
      { tag: 257, type: T_LONG, value: height },                       // ImageLength
      { tag: 258, type: T_SHORT, value: 32 },                          // BitsPerSample
      { tag: 259, type: T_SHORT, value: compression },                 // Compression
      { tag: 262, type: T_SHORT, value: 1 },                           // PhotometricInterpretation MinIsBlack
      { tag: 277, type: T_SHORT, value: 1 },                           // SamplesPerPixel
      { tag: 278, type: T_SHORT, value: rowsPerStrip },                // RowsPerStrip
      { tag: 284, type: T_SHORT, value: 1 },                           // PlanarConfiguration Contig
      { tag: 317, type: T_SHORT, value: 1 },                           // Predictor = 1 (none; deflate alone)
      { tag: 339, type: T_SHORT, value: 3 },                           // SampleFormat = IEEE float
      { tag: 273, type: T_LONG, blob: longArrForStrips(strips, 0) },  // StripOffsets
      { tag: 279, type: T_LONG, blob: longArrForStrips(strips, 1) },  // StripByteCounts
      { tag: 33550, type: T_DOUBLE, blob: doubles(pixelScale) },      // ModelPixelScale
      { tag: 33922, type: T_DOUBLE, blob: doubles(tiePoint) },        // ModelTiePoint
      { tag: 34735, type: T_SHORT, blob: shorts(geoKeys) },            // GeoKeyDirectory
      { tag: 42112, type: T_ASCII, blob: xmlBytes },                   // GDAL_METADATA
      { tag: 42113, type: T_ASCII, blob: nodataBytes }                 // GDAL_NODATA
    ];
    if (descBytes) specs.push({ tag: 270, type: T_ASCII, blob: descBytes });   // ImageDescription

    // count + inline flag per spec.  The rule libtiff/GDAL apply is byte-size based: a value whose
    // count*type_size is <= 4 bytes lives IN the IFD entry, and a reader will not follow an offset
    // for it - so a 4-byte ASCII like GDAL_NODATA "nan\0" must be inlined, not pointed at.
    specs.forEach(function (sp) {
      if (sp.blob) {
        sp.count = sp.blob.length / SIZEOF[sp.type];
        sp.inline = sp.blob.length <= 4;
      } else { sp.count = 1; sp.inline = true; }
    });

    // ---- layout: header | IFD | extras (4-aligned) | strip data -----------------------------
    specs.sort(function (a, b) { return a.tag - b.tag; });            // TIFF requires ascending tags
    var ifdStart = 8;
    var ifdLen = 2 + specs.length * 12 + 4;
    var extrasAt = ifdStart + ifdLen, cur = align4(extrasAt);
    for (var q = 0; q < specs.length; q++) {
      var sp = specs[q];
      if (sp.inline) continue;
      sp.offset = cur;
      cur += sp.blob.length;
      if (sp.blob.length % 2) cur += 1;                              // word-align the next block
    }
    var dataStart = align4(cur);
    var at = dataStart, offs = [], lens = [];
    for (var w = 0; w < strips.length; w++) {
      offs.push(at); lens.push(strips[w].bytes.length);
      at += align4(strips[w].bytes.length);
    }
    var total = at;
    // strip offsets/counts must be written now that offsets are known
    for (var f = 0; f < specs.length; f++) {
      if (specs[f].tag === 273) specs[f].blob = longs(offs);
      if (specs[f].tag === 279) specs[f].blob = longs(lens);
      specs[f].offset = specs[f].inline ? 0 : specs[f].offset;
    }
    // the two patched arrays are the same size as the placeholders, so the layout still holds

    var out = zeros(total), dv = new DataView(out.buffer);
    out[0] = 0x49; out[1] = 0x49;                                    // "II"
    dv.setUint16(2, 42, true);
    dv.setUint32(4, ifdStart, true);
    dv.setUint16(ifdStart, specs.length, true);
    for (var i2 = 0; i2 < specs.length; i2++) {
      var t = specs[i2], e = ifdStart + 2 + i2 * 12;
      dv.setUint16(e, t.tag, true); dv.setUint16(e + 2, t.type, true);
      dv.setUint32(e + 4, t.count >>> 0, true);
      if (t.inline) {
        if (t.blob) { for (var b3 = 0; b3 < 4; b3++) out[e + 8 + b3] = t.blob[b3] || 0; }
        else if (t.type === T_SHORT) { dv.setUint16(e + 8, t.value & 0xffff, true); }
        else dv.setUint32(e + 8, t.value >>> 0, true);
      } else {
        dv.setUint32(e + 8, t.offset >>> 0, true);
        out.set(t.blob, t.offset);
      }
    }
    dv.setUint32(ifdStart + 2 + specs.length * 12, 0, true);         // no next IFD
    for (var s2 = 0; s2 < strips.length; s2++) out.set(strips[s2].bytes, offs[s2]);

    return {
      bytes: out,
      layout: {
        width: width, height: height, nStrips: strips.length, rowsPerStrip: rowsPerStrip,
        compression: compression, deflated: compressedAny, ifdStart: ifdStart, ifdBytes: ifdLen,
        extrasBytes: cur - extrasAt, dataStart: dataStart, totalBytes: total,
        rawFieldBytes: pixels.length, containerBytes: total,
        strippedBytes: lens.reduce(function (a, b) { return a + b; }, 0)
      }
    };

    function longArrForStrips(list, which) {                          // placeholder; same byte size
      return longs(list.map(function (_, idx) { return idx; }));
    }
  }

  // -------------------------------------------------------------- minimal reader (self-check)
  function parseGeoTIFF(bytes) {
    var view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
    var mark = view.getUint16(0, true);
    if (mark !== 0x4949 && mark !== 0x4d4d) throw new Error('not a TIFF: bad byte-order mark 0x' + mark.toString(16));
    var le = mark === 0x4949;
    if (view.getUint16(2, le) !== 42) throw new Error('not a classic TIFF: version != 42');
    var ifd = view.getUint32(4, le);
    var n = view.getUint16(ifd, le);
    var tags = {};
    for (var i = 0; i < n; i++) {
      var e = ifd + 2 + i * 12;
      var tag = view.getUint16(e, le), type = view.getUint16(e + 2, le), count = view.getUint32(e + 4, le);
      var size = (SIZEOF[type] || 1) * count;
      var inlineStore = size <= 4;
      tags[tag] = {
        type: type, count: count, size: size, inline: inlineStore,
        offset: inlineStore ? e + 8 : view.getUint32(e + 8, le)
      };
    }
    function asShorts(t) { var o = []; for (var k = 0; k < t.count; k++) o.push(view.getUint16(t.offset + 2 * k, le)); return o; }
    function asLongs(t) { var o = []; for (var k = 0; k < t.count; k++) o.push(view.getUint32(t.offset + 4 * k, le)); return o; }
    function asDoubles(t) { var o = []; for (var k = 0; k < t.count; k++) o.push(view.getFloat64(t.offset + 8 * k, le)); return o; }
    function asAscii(t) { var s = ''; for (var k = 0; k < t.count; k++) { var c = view.getUint8(t.offset + k); if (!c) break; s += String.fromCharCode(c); } return s; }
    function one(t) { if (!t) return null; return t.type === T_SHORT ? view.getUint16(t.offset, le) : view.getUint32(t.offset, le); }

    var out = {
      littleEndian: le, tags: tags, ifdEntries: n,
      width: one(tags[256]), height: one(tags[257]),
      bitsPerSample: one(tags[258]), compression: one(tags[259]),
      photometric: one(tags[262]), samplesPerPixel: one(tags[277]),
      rowsPerStrip: one(tags[278]), planar: one(tags[284]),
      predictor: tags[317] ? one(tags[317]) : 1,
      sampleFormat: one(tags[339]),
      imageDescription: tags[270] ? asAscii(tags[270]) : null,
      stripOffsets: asLongs(tags[273]), stripByteCounts: asLongs(tags[279]),
      modelPixelScale: tags[33550] ? asDoubles(tags[33550]) : null,
      modelTiePoint: tags[33922] ? asDoubles(tags[33922]) : null,
      geoKeys: tags[34735] ? asShorts(tags[34735]) : null,
      nodata: tags[42113] ? asAscii(tags[42113]) : null,
      gdalMetadata: tags[42112] ? asAscii(tags[42112]) : null
    };
    out.epsg = 0; out.modelType = 0; out.rasterType = 0;
    if (out.geoKeys) {
      out.keyVersion = out.geoKeys[0]; out.keyRevision = out.geoKeys[1]; out.keyMinor = out.geoKeys[2];
      out.nKeys = out.geoKeys[3];
      for (var g = 4; g + 3 < out.geoKeys.length; g += 4) {
        if (out.geoKeys[g] === 3072) out.epsg = out.geoKeys[g + 3];
        if (out.geoKeys[g] === 1024) out.modelType = out.geoKeys[g + 3];
        if (out.geoKeys[g] === 1025) out.rasterType = out.geoKeys[g + 3];
      }
    }
    out.geoTransform = (out.modelPixelScale && out.modelTiePoint) ? [
      out.modelPixelScale[0], 0, out.modelTiePoint[3],
      0, -out.modelPixelScale[1], out.modelTiePoint[4]
    ] : null;
    return out;
  }

  /** decompress + reassemble the raster from the file's own tags */
  async function readField(bytes, info) {
    var u8 = new Uint8Array(bytes.buffer, bytes.byteOffset, bytes.byteLength);
    var out = new Float32Array(info.width * info.height);
    var at = 0;                                          // next pixel index to write
    for (var s = 0; s < info.stripOffsets.length; s++) {
      var rows = Math.min(info.rowsPerStrip, info.height - s * info.rowsPerStrip);
      var chunk = u8.subarray(info.stripOffsets[s], info.stripOffsets[s] + info.stripByteCounts[s]);
      var raw = info.compression === COMPRESSION_NONE ? chunk : await inflateZlib(chunk, true);
      if (raw.length !== rows * info.width * 4) {
        throw new Error('strip ' + s + ' is ' + raw.length + ' B, expected ' + (rows * info.width * 4));
      }
      if (at + rows * info.width > out.length) throw new Error('strip ' + s + ' overruns the grid');
      out.set(new Float32Array(raw.buffer, raw.byteOffset, raw.byteLength / 4), at);
      at += rows * info.width;
    }
    if (at !== out.length) throw new Error('strips cover ' + at + ' px of a ' + out.length + ' px grid');
    return out;
  }

  // -------------------------------------------------------------- checksums
  async function sha256Hex(bytes) {
    if (global.crypto && global.crypto.subtle) {
      var d = await global.crypto.subtle.digest('SHA-256', bytes);
      var o = new Uint8Array(d), s = '';
      for (var i = 0; i < o.length; i++) s += o[i].toString(16).padStart(2, '0');
      return s;
    }
    throw new Error('no WebCrypto in this runtime (crypto.subtle)');
  }
  async function sha256OfFloat32(field) {
    return sha256Hex(new Uint8Array(field.buffer, field.byteOffset, field.byteLength));
  }

  var CRC_TABLE = (function () {
    var t = new Uint32Array(256);
    for (var n = 0; n < 256; n++) {
      var c = n;
      for (var k = 0; k < 8; k++) c = (c & 1) ? (0xEDB88320 ^ (c >>> 1)) : (c >>> 1);
      t[n] = c >>> 0;
    }
    return t;
  })();
  function crc32(u8) {
    var c = 0xFFFFFFFF;
    for (var i = 0; i < u8.length; i++) c = CRC_TABLE[(c ^ u8[i]) & 0xff] ^ (c >>> 8);
    return (c ^ 0xFFFFFFFF) >>> 0;
  }
  function dosDateTime(ms) {
    var d = new Date(ms);
    var time = (d.getUTCHours() << 11) | (d.getUTCMinutes() << 5) | (d.getUTCSeconds() / 2 | 0);
    var yr = Math.max(1980, d.getUTCFullYear());
    var date = ((yr - 1980) << 9) | ((d.getUTCMonth() + 1) << 5) | d.getUTCDate();
    return { time: time & 0xffff, date: date & 0xffff };
  }

  /** A .zip holding exactly one stored member: the platform accepts "a .zip file containing a
   *  single GeoTIFF", and method 0 (stored) is the simplest container that claim permits. */
  function buildZip(entries) {
    // Fixed DOS stamp (2020-01-01T00:00:00Z) unless the caller overrides it, matching
    // scripts/package_submission.py's FIXED_DATE_TIME. Not Date.now(): the container should be
    // reproducible, and "zip the same raster twice" should not depend on the wall clock.
    var stamp = dosDateTime(entries[0] && entries[0].date ? entries[0].date
                              : Date.UTC(2020, 0, 1, 0, 0, 0));
    var parts = [], central = [], at = 0;
    for (var e = 0; e < entries.length; e++) {
      var nm = enc(entries[e].name), data = entries[e].data, crc = crc32(data);
      parts.push(cat([u32(0x04034b50), u16(20), u16(0), u16(0), u16(stamp.time), u16(stamp.date),
                      u32(crc), u32(data.length), u32(data.length), u16(nm.length), u16(0), nm, data]));
      central.push(cat([u32(0x02014b50), u16(20), u16(20), u16(0), u16(0), u16(stamp.time), u16(stamp.date),
                        u32(crc), u32(data.length), u32(data.length), u16(nm.length), u16(0), u16(0),
                        u16(0), u16(0), u32(0), u32(at), nm]));
      at += 30 + nm.length + data.length;
    }
    var cd = cat(central);
    var eocd = cat([u32(0x06054b50), u16(0), u16(0), u16(entries.length), u16(entries.length),
                    u32(cd.length), u32(at), u16(0)]);
    return cat([].concat(parts, [cd, eocd]));
  }

  // -------------------------------------------------------------- top level
  /** meta: parsed docs/submission_meta.json · fieldBytes: the raw docs/submission_field.bin */
  async function generateFromPayload(opts) {
    var meta = opts.meta, fieldBytes = opts.fieldBytes;
    var grid = meta.grid, enc2 = meta.encoding;
    if (enc2.format !== 'gems-rle-v1') {
      throw new Error('payload encoding is "' + enc2.format + '"; this writer implements gems-rle-v1 only');
    }
    if (grid.dtype !== 'float32' || grid.count !== 1) {
      throw new Error('payload is not a single-band float32 grid (dtype ' + grid.dtype + ', count ' + grid.count + ')');
    }
    if (meta.blob && meta.blob.bytes && meta.blob.bytes !== fieldBytes.length) {
      throw new Error('field blob is ' + fieldBytes.length + ' B, manifest says ' + meta.blob.bytes + ' B');
    }
    var dec = decodeRuns(fieldBytes, grid.width, grid.height);
    var field = dec.field, u32v = new Uint32Array(field.buffer);

    var checks = [];
    function check(name, ok, measured) { checks.push({ name: name, ok: !!ok, measured: String(measured) }); }

    var onePx = 0, nanPx = 0, zeroPx = 0, other = 0;
    for (var i = 0; i < field.length; i++) {
      var v = field[i];
      if (v !== v) nanPx++; else if (v === 1) onePx++; else if (v === 0) zeroPx++; else other++;
    }
    check('run stream tiles the grid', dec.runs > 0, dec.runs.toLocaleString() + ' runs → ' + field.length.toLocaleString() + ' px');
    check('pixels at 1.0 equal the pinned count', onePx === meta.field.one_px, onePx.toLocaleString() + ' vs ' + meta.field.one_px.toLocaleString());
    check('pixels at 0.0 equal the pinned count', zeroPx === meta.field.zero_px, zeroPx.toLocaleString() + ' vs ' + meta.field.zero_px.toLocaleString());
    check('NaN pixels equal the pinned count', nanPx === meta.field.nan_px, nanPx.toLocaleString() + ' vs ' + meta.field.nan_px.toLocaleString());
    check('no value outside {0, 1, NaN}', other === 0, other + ' unexpected px');
    check('pinned finite range lies in [0,1]', meta.field.min >= 0 && meta.field.max <= 1, 'min ' + meta.field.min + ' max ' + meta.field.max);
    if (meta.blob && meta.blob.sha256) {
      var blobHex = await sha256Hex(fieldBytes);
      check('field blob sha256 equals the manifest', blobHex === meta.blob.sha256, blobHex.slice(0, 16) + '…');
    }
    var fieldHex = await sha256OfFloat32(field);
    check('decoded field hashes to the pinned float32 buffer', fieldHex === meta.field.float32_sha256,
          fieldHex.slice(0, 16) + '… vs ' + String(meta.field.float32_sha256).slice(0, 16) + '…');

    var built = await buildGeoTIFF(field, grid, { compression: opts.compression, rowsPerStrip: opts.rowsPerStrip });
    var bytes = built.bytes;

    var info = parseGeoTIFF(bytes);
    check('re-read: single band, float32, IEEE sample format',
          info.samplesPerPixel === 1 && info.bitsPerSample === 32 && info.sampleFormat === 3,
          'Samples ' + info.samplesPerPixel + ' · Bits ' + info.bitsPerSample + ' · Format ' + info.sampleFormat);
    check('re-read: grid matches the artifact', info.width === grid.width && info.height === grid.height,
          info.width.toLocaleString() + ' × ' + info.height.toLocaleString());
    check('re-read: GeoKeyDirectory carries EPSG:' + grid.epsg, info.epsg === grid.epsg,
          'EPSG:' + info.epsg + ' · GTModelType ' + info.modelType + ' · GTRasterType ' + info.rasterType);
    var ps = grid.pixelscale || [grid.res[0], Math.abs(grid.res[1]), 0];
    var scaleOk = info.modelPixelScale && Math.abs(info.modelPixelScale[0] - ps[0]) < 1e-9 &&
                  Math.abs(info.modelPixelScale[1] - ps[1]) < 1e-9;
    check('re-read: ModelPixelScale carries the 100 m resolution', scaleOk,
          info.modelPixelScale ? info.modelPixelScale.join(', ') : 'absent');
    var tp = info.modelTiePoint, ox = grid.transform[2], oy = grid.transform[5];
    var tieOk = tp && tp[0] === 0 && tp[1] === 0 && Math.abs(tp[3] - ox) < 1e-6 && Math.abs(tp[4] - oy) < 1e-6;
    check('re-read: ModelTiePoint carries the upper-left origin', tieOk,
          tp ? 'x ' + tp[3] + ' · y ' + tp[4] : 'absent');
    check('re-read: GDAL_NODATA is "nan"', info.nodata === 'nan', JSON.stringify(info.nodata));
    var rt = await readField(bytes, info);
    var rt32 = new Uint32Array(rt.buffer);
    var diff = 0;
    for (var j = 0; j < rt.length; j++) if (rt32[j] !== u32v[j]) diff++;
    check('re-read: every float32 bit survives the container', diff === 0,
          diff + ' px differ of ' + rt.length.toLocaleString());
    check('re-read: strip table is self-consistent',
          info.stripOffsets.length === info.stripByteCounts.length &&
          info.stripOffsets.length === Math.ceil(info.height / info.rowsPerStrip),
          info.stripOffsets.length + ' strips · ' + built.layout.totalBytes.toLocaleString() + ' B');
    check('re-read: GeoTIFF metadata round-trips (AREA_OR_POINT)', /AREA_OR_POINT/.test(info.gdalMetadata || ''),
          (info.gdalMetadata || '').replace(/\s+/g, ' ').trim().slice(0, 64));

    var failed = checks.filter(function (c) { return !c.ok; });
    var hex = await sha256Hex(bytes);
    return {
      bytes: bytes, sha256: hex, layout: built.layout, info: info, checks: checks,
      ok: failed.length === 0, failed: failed.map(function (c) { return c.name; }),
      field: { one_px: onePx, zero_px: zeroPx, nan_px: nanPx, runs: dec.runs },
      compression: built.layout.compression === COMPRESSION_DEFLATE ? 'deflate (zlib-wrapped, per strip — tag 8, as GDAL writes)' : 'none (uncompressed strips)'
    };
  }

  var api = {
    decodeRuns: decodeRuns, buildGeoTIFF: buildGeoTIFF, parseGeoTIFF: parseGeoTIFF,
    readField: readField, generateFromPayload: generateFromPayload, buildZip: buildZip,
    crc32: crc32, sha256Hex: sha256Hex, hasStreams: hasStreams,
    inflateZlib: inflateZlib,
    COMPRESSION_NONE: COMPRESSION_NONE, COMPRESSION_DEFLATE: COMPRESSION_DEFLATE,
    COMPRESSION_DEFLATE_ALT: COMPRESSION_DEFLATE_ALT
  };

  if (typeof module !== 'undefined' && module.exports) {
    module.exports = api;
    if (require.main === module) {
      var fs = require('fs');
      (async function () {
        var args = process.argv.slice(2);
        // Positional = the output paths. An option that takes a value (--rows-per-strip N) must
        // swallow its argument, or "64" lands in `pos` and silently becomes a fourth output file.
        var OPTS_WITH_VALUE = { '--rows-per-strip': 1 };
        var pos = [], flags = [];
        for (var ai = 0; ai < args.length; ai++) {
          var a = args[ai];
          if (a.slice(0, 2) === '--') {
            flags.push(a);
            var nv = OPTS_WITH_VALUE[a] || 0;
            for (var ni = 0; ni < nv; ni++) { ai++; flags.push(args[ai]); }
          } else { pos.push(a); }
        }
        if (pos.length < 3) {
          process.stderr.write('usage: node docs/geotiff_writer.js <meta.json> <field.bin> <out.tif> ' +
                               '[out.zip] [--no-deflate] [--rows-per-strip N]\n');
          process.exit(2);
        }
        var meta = JSON.parse(fs.readFileSync(pos[0], 'utf8'));
        var blob = new Uint8Array(fs.readFileSync(pos[1]));
        // the VALUE of an option is the next entry of argv, not of `flags` (only entries that
        // start with -- land in flags, so flag+1 there is undefined - which silently made every
        // CLI run use the default strip height and every strip-size test a no-op).
        var rps;
        var ri = flags.indexOf('--rows-per-strip');
        if (ri >= 0) {
          rps = parseInt(flags[ri + 1], 10);
          if (!(rps > 0)) throw new Error('--rows-per-strip needs a positive integer, got ' + flags[ri + 1]);
        }
        var res = await generateFromPayload({
          meta: meta, fieldBytes: blob,
          compression: flags.indexOf('--no-deflate') >= 0 ? 'none' : 'deflate',
          rowsPerStrip: rps
        });
        if (!res.ok) {
          // Write nothing. A file left on disk by a failed build is the exact artefact someone
          // uploads by accident three days later; the browser path already withholds the download.
          process.stderr.write('REFUSED ' + res.failed.join('; ') + '\n');
          process.stdout.write(JSON.stringify({ ok: false, failed: res.failed, checks: res.checks.map(
            function (c) { return { name: c.name, ok: c.ok, measured: c.measured }; }) }, null, 1) + '\n');
          process.exit(1);
        }
        fs.writeFileSync(pos[2], Buffer.from(res.bytes));
        if (pos[3]) {
          var z = buildZip([{ name: 'submission.tif', data: res.bytes }]);   // fixed 2020-01-01 stamp
          fs.writeFileSync(pos[3], Buffer.from(z));
        }
        process.stdout.write(JSON.stringify({
          ok: res.ok, failed: res.failed, sha256: res.sha256, bytes: res.bytes.length,
          compression: res.compression, layout: res.layout, field: res.field,
          checks: res.checks.map(function (c) { return { name: c.name, ok: c.ok, measured: c.measured }; })
        }, null, 1) + '\n');
        process.exit(res.ok ? 0 : 1);
      })().catch(function (err) { process.stderr.write('ERROR ' + ((err && err.stack) || err) + '\n'); process.exit(3); });
    }
  } else {
    global.GemsGeoTIFF = api;
  }
})(typeof window !== 'undefined' ? window : globalThis);
