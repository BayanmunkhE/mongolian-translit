// Browser port of the torch-free tagger path (translit/onnx_backend.py, translit/mixed.py, translit/keyboard.py).
// Everything runs on the visitor's device with onnxruntime-web; no server. Assets come from tools/build_site.py.
/* global ort */
(function () {
  'use strict';
  const ORT_VERSION = '1.20.1';
  const ASSETS = 'assets/';
  const UNK = '<unk>';
  const MAX_BYTES = 200;
  const CYRILLIC = /[Ѐ-ӿ]/;
  const LATIN = /[A-Za-z]/;
  // A run of whole words without any Cyrillic letter (translit/mixed.py RUN).
  const RUN = /(?:^|(?<=\s))[^\sЀ-ӿ]+(?:\s+[^\sЀ-ӿ]+)*(?=\s|$)/g;
  const utf8 = new TextEncoder();

  let session = null, vocab = null, dict = {}, bigrams = {}, modelInfo = {};
  const cache = new Map();

  function encode(text) {
    const bytes = utf8.encode(text);
    const ids = new BigInt64Array(bytes.length + 1);
    for (let i = 0; i < bytes.length; i++) ids[i] = BigInt(bytes[i] + 3);
    ids[bytes.length] = 1n;  // </s>
    return ids;
  }

  function decode(src, pred) {
    let out = '', pos = 0;
    for (const ch of src) {
      const lab = pos < pred.length ? vocab[pred[pos]] : UNK;
      out += (lab === UNK || lab === undefined) ? ch : lab;
      pos += utf8.encode(ch).length;
    }
    return out;
  }

  function chunks(text) {
    if (utf8.encode(text).length <= MAX_BYTES) return [text];
    const out = []; let cur = '';
    for (const w of text.split(/\s+/).filter(Boolean)) {
      const cand = cur ? cur + ' ' + w : w;
      if (cur && utf8.encode(cand).length > MAX_BYTES) { out.push(cur); cur = w; } else cur = cand;
    }
    if (cur) out.push(cur);
    return out;
  }

  async function runTagger(text) {
    const ids = encode(text);
    const mask = new BigInt64Array(ids.length).fill(1n);
    const feeds = {
      input_ids: new ort.Tensor('int64', ids, [1, ids.length]),
      attention_mask: new ort.Tensor('int64', mask, [1, ids.length]),
    };
    const res = await session.run(feeds);
    const data = res[session.outputNames[0]].data;
    return Array.from(data, Number);
  }

  async function convertLine(line) {
    if (cache.has(line)) return cache.get(line);
    const parts = [];
    for (const c of chunks(line)) parts.push(decode(c, await runTagger(c)));
    const out = parts.join(' ');
    if (cache.size > 4000) cache.clear();
    cache.set(line, out);
    return out;
  }

  // OnnxTaggerTransliterator.transliterate: line by line, empty lines kept.
  async function transliterate(text) {
    text = text.trim();
    if (!text) return '';
    const lines = text.split('\n');
    const out = [];
    for (const l of lines) out.push(l.trim() ? await convertLine(l) : '');
    return out.join('\n');
  }

  // translit/mixed.py convert_mixed: Cyrillic words stay as typed, only Latin runs go to the model.
  async function convertMixed(text) {
    if (!CYRILLIC.test(text)) return transliterate(text);
    let out = '', last = 0;
    for (const m of text.matchAll(RUN)) {
      const span = m[0];
      out += text.slice(last, m.index);
      if (LATIN.test(span)) {
        let conv = await transliterate(span);
        if (m.index > 0 && isLower(span[0]) && isUpper(conv[0])) conv = conv[0].toLowerCase() + conv.slice(1);
        out += conv;
      } else out += span;
      last = m.index + span.length;
    }
    return out + text.slice(last);
  }

  function isLower(c) { return !!c && c === c.toLowerCase() && c !== c.toUpperCase(); }
  function isUpper(c) { return !!c && c === c.toUpperCase() && c !== c.toLowerCase(); }

  // translit/keyboard.py helpers
  function matchCase(latin, word) {
    if (latin && isLower(latin[0]) && word && isUpper(word[0])) {
      if (word.length > 1 && word === word.toUpperCase()) return word;
      return word[0].toLowerCase() + word.slice(1);
    }
    if (latin && isUpper(latin[0]) && word && isLower(word[0])) return word[0].toUpperCase() + word.slice(1);
    return word;
  }

  function rankDictionary(cands, follow) {
    if (!follow) return cands.slice();
    return cands.map((w, i) => [w, i]).sort((a, b) => ((follow[b[0].toLowerCase()] || 0) - (follow[a[0].toLowerCase()] || 0)) || (a[1] - b[1])).map(x => x[0]);
  }

  function tailAfterPrefix(full, prefixConv, nWords) {
    if (prefixConv && full.toLowerCase().startsWith(prefixConv.toLowerCase())) {
      const rest = full.slice(prefixConv.length).trim();
      if (rest) return rest;
    }
    const words = full.split(/\s+/).filter(Boolean);
    return words.length ? words.slice(-nWords).join(' ') : full;
  }

  function merge(groups, latin) {
    const out = [], seen = new Set();
    for (const g of groups) for (let w of g) {
      w = matchCase(latin, w.trim());
      if (w && !seen.has(w.toLowerCase())) { seen.add(w.toLowerCase()); out.push(w); }
    }
    if (!seen.has(latin.toLowerCase())) out.push(latin);
    return out;
  }

  // KeyboardSuggester.suggest (without the seq2seq fallback, which is too large for a web page).
  async function suggest(word, context, prevCyr, n) {
    word = (word || '').trim();
    if (!word) return { candidates: [], source: null };
    context = (context || []).filter(c => c.trim()).slice(-2);
    const follow = prevCyr ? bigrams[prevCyr.toLowerCase()] : null;
    const fromDict = rankDictionary(dict[word.toLowerCase()] || [], follow);
    let full = await transliterate(context.concat([word]).join(' '));
    if (context.length) full = tailAfterPrefix(full, await transliterate(context.join(' ')), 1);
    const cands = merge([[full], fromDict], word);
    return { candidates: n ? cands.slice(0, n) : cands, source: 'tagger' };
  }

  async function fetchWithProgress(url, onProgress) {
    const r = await fetch(url);
    if (!r.ok) throw new Error(url + ': HTTP ' + r.status);
    const total = +r.headers.get('Content-Length') || 0;
    if (!r.body || !total) { const b = await r.arrayBuffer(); onProgress && onProgress(1); return b; }
    const reader = r.body.getReader(); const parts = []; let got = 0;
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      parts.push(value); got += value.length;
      onProgress && onProgress(Math.min(1, got / total));
    }
    const buf = new Uint8Array(got); let o = 0;
    for (const p of parts) { buf.set(p, o); o += p.length; }
    return buf.buffer;
  }

  async function getJSON(url) { const r = await fetch(url); if (!r.ok) throw new Error(url + ': HTTP ' + r.status); return r.json(); }

  // opts.keyboard: also load the dictionary and bigrams (the message page does not need them).
  async function load(opts, onProgress) {
    opts = opts || {};
    ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@' + ORT_VERSION + '/dist/';
    ort.env.wasm.numThreads = 1;  // GitHub Pages is not cross-origin isolated
    const info = await getJSON(ASSETS + 'model.json');
    modelInfo = info;
    const jobs = [getJSON(ASSETS + 'vocab.json').then(v => { vocab = v; })];
    if (opts.keyboard) {
      jobs.push(getJSON(ASSETS + 'dict.json').then(d => { dict = d; }));
      jobs.push(getJSON(ASSETS + 'bigrams.json').then(b => { bigrams = b; }));
    }
    const model = await fetchWithProgress(ASSETS + info.file, onProgress);
    await Promise.all(jobs);
    session = await ort.InferenceSession.create(model, { executionProviders: ['wasm'], graphOptimizationLevel: 'all' });
    return info;
  }

  window.MnTranslit = { load, transliterate, convertMixed, suggest, info: () => modelInfo,
                        _internals: { encode, decode, chunks, matchCase, rankDictionary, tailAfterPrefix, merge } };
})();
