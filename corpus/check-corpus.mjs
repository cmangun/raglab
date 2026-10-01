#!/usr/bin/env node
// Checks the generated corpus and golden set. It reads only the generated
// files, not the generator, so it catches a generator that is wrong as well as
// one that is out of date. Exits non-zero on any failure.
//
// Usage: node corpus/check-corpus.mjs

import { readFileSync, readdirSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const ROOT = dirname(fileURLToPath(import.meta.url));
const json = (rel) => JSON.parse(readFileSync(join(ROOT, rel), 'utf8'));

let failures = 0;
const results = [];
function check(name, ok, detail = '') {
	results.push({ name, ok, detail });
	if (!ok) failures++;
}

// ------------------------------------------------------------ load documents
const slug = (s) => s.toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/^-|-$/g, '');
const docs = {};
for (const f of readdirSync(join(ROOT, 'documents')).filter((f) => f.endsWith('.md'))) {
	const raw = readFileSync(join(ROOT, 'documents', f), 'utf8');
	const m = raw.match(/^---\n([\s\S]*?)\n---\n([\s\S]*)$/);
	if (!m) { check(`front matter parses: ${f}`, false); continue; }
	const meta = {};
	for (const line of m[1].split('\n')) {
		const i = line.indexOf(':');
		meta[line.slice(0, i).trim()] = line.slice(i + 1).trim();
	}
	meta.groups = meta.groups.replace(/[\[\]]/g, '').split(',').map((s) => s.trim());
	const sections = {};
	for (const part of m[2].split(/\n## /).slice(1)) {
		const nl = part.indexOf('\n');
		sections[slug(part.slice(0, nl))] = part.slice(nl + 1).trim();
	}
	docs[meta.id] = { ...meta, sections, raw };
}
const sites = json('data/site.json');
const products = json('data/product.json');
const batches = json('data/batch.json');
const deviations = json('data/deviation.json');
const golden = json('golden.json');
const manifest = json('manifest.json');
const all = Object.values(docs);

// ------------------------------------------------------------------- counts
const byType = {};
for (const d of all) byType[d.type] = (byType[d.type] || 0) + 1;
check('60 documents', all.length === 60, `found ${all.length}`);
check('document types 20/12/9/6/8/5',
	byType.sop === 20 && byType.spec === 12 && byType.deviation_report === 9 && byType.change_control === 6 && byType.commercial_memo === 8 && byType.audit_finding === 5,
	JSON.stringify(byType));
const both = all.filter((d) => d.groups.length === 2).length;
const comOnly = all.filter((d) => d.groups.join() === 'commercial').length;
const quaOnly = all.filter((d) => d.groups.join() === 'quality').length;
check('access lists 47 both / 8 commercial / 5 quality', both === 47 && comOnly === 8 && quaOnly === 5, `${both}/${comOnly}/${quaOnly}`);
const sectionCount = all.reduce((n, d) => n + Object.keys(d.sections).length, 0);
check('section count matches manifest', sectionCount === manifest.sections, `${sectionCount} vs ${manifest.sections}`);
check('every document carries the synthetic notice', all.every((d) => d.raw.includes('Synthetic document.') && d.synthetic === 'true'));

// --------------------------------------------------------------- versioning
const superseded = all.filter((d) => d.status === 'superseded');
check('4 superseded documents', superseded.length === 4, `found ${superseded.length}`);
check('each superseded document points at a current one',
	superseded.every((d) => docs[d.superseded_by] && docs[d.superseded_by].status === 'current'));

// ------------------------------------------------------- relational integrity
const ids = (rows) => new Set(rows.map((r) => r.id));
const siteIds = ids(sites), productIds = ids(products), batchIds = ids(batches);
const unique = (rows) => ids(rows).size === rows.length;
check('row ids are unique', unique(sites) && unique(products) && unique(batches) && unique(deviations));
check('batches reference real products and sites', batches.every((b) => productIds.has(b.product_id) && siteIds.has(b.site_id)));
check('deviations reference real batches, products and sites',
	deviations.every((d) => batchIds.has(d.batch_id) && productIds.has(d.product_id) && siteIds.has(d.site_id)));
const batchById = Object.fromEntries(batches.map((b) => [b.id, b]));
check('deviation product and site match its batch',
	deviations.every((d) => batchById[d.batch_id].product_id === d.product_id && batchById[d.batch_id].site_id === d.site_id));
check('deviations open on or after the batch date and inside 2025',
	deviations.every((d) => d.opened_date >= batchById[d.batch_id].mfg_date && d.opened_date <= '2025-12-31'));
check('deviations close on or after they open', deviations.every((d) => d.closed_date === null || d.closed_date >= d.opened_date));
check('linked change controls exist as documents', deviations.every((d) => d.linked_change_id === null || docs[d.linked_change_id]));

// ------------------------------------------------------------ confidentiality
// Names that must never appear: real employers, clients and systems from the
// case studies. The list holds public names only.
const banned = ['abbott', 'architect', 'alinity', 'idamart', 'bdaa', 'exadata', 'pfizer', 'novartis', 'sanofi', 'publicis', 'ipg', 'lilly', 'trulicity', 'syneos'];
const corpusText = (all.map((d) => d.raw).join('\n') + JSON.stringify([sites, products, golden])).toLowerCase();
const hits = banned.filter((w) => new RegExp(`\\b${w}\\b`).test(corpusText));
check('no real employer, client or system names', hits.length === 0, hits.join(', '));

// ---------------------------------------------------------------- golden set
const typeCounts = {};
for (const q of golden) typeCounts[q.type] = (typeCounts[q.type] || 0) + 1;
const want = { lookup: 8, multi_hop: 6, numeric: 6, relationship: 6, table: 6, superseded: 4, out_of_scope: 3, forbidden: 3 };
check('42 questions with the planned type counts', golden.length === 42 && Object.entries(want).every(([t, n]) => typeCounts[t] === n), JSON.stringify(typeCounts));

const sectionText = (ref) => {
	const [docId, s] = ref.split('#');
	return docs[docId]?.sections[s];
};
const numbers = (s) => (s.match(/-?\d+(?:\.\d+)?/g) || []);

for (const q of golden) {
	const sectionRefs = q.supporting_refs.filter((r) => r.kind === 'section');
	const missing = sectionRefs.filter((r) => sectionText(r.ref) === undefined);
	check(`${q.id} supporting sections exist`, missing.length === 0, missing.map((r) => r.ref).join(', '));
	if (missing.length) continue;
	const supportDocs = sectionRefs.map((r) => docs[r.ref.split('#')[0]]);
	const text = sectionRefs.map((r) => sectionText(r.ref)).join('\n');

	if (q.type === 'forbidden') {
		check(`${q.id} evidence is closed to the asking group`, supportDocs.length > 0 && supportDocs.every((d) => !d.groups.includes(q.ask_as_group)));
		check(`${q.id} expects access_denied`, q.expected_verdict === 'access_denied');
	} else if (q.type === 'out_of_scope') {
		check(`${q.id} has no evidence and expects insufficient_evidence`, q.supporting_refs.length === 0 && q.expected_verdict === 'insufficient_evidence');
	} else {
		check(`${q.id} expects an answer`, q.expected_verdict === 'answered');
		check(`${q.id} evidence is open to the asking group`, supportDocs.every((d) => d.groups.includes(q.ask_as_group)));
		check(`${q.id} evidence is current`, supportDocs.every((d) => d.status === 'current'));
	}

	if (['lookup', 'table', 'superseded', 'multi_hop'].includes(q.type)) {
		// the leading figure of the expected answer must be present in the evidence
		const lead = numbers(q.expected_answer.replace(/\((?:SOP|version)[^)]*\)/g, '').replace(/SOP-\d+|P-\d+|version \d+/g, ''));
		const ok = lead.length > 0 && lead.every((n) => text.includes(n));
		check(`${q.id} expected figures appear in the evidence`, ok, `${lead.join(', ')}`);
	}
	if (q.expected_verdict === 'answered') {
		const hay = (text + ' ' + q.expected_answer).toLowerCase();
		const present = (f) => (Array.isArray(f) ? f.some(present) : hay.includes(String(f).toLowerCase()));
		check(`${q.id} has key facts and they appear in its evidence or answer`, q.key_facts.length > 0 && q.key_facts.every(present), JSON.stringify(q.key_facts));
	}
	if (q.type === 'forbidden') {
		check(`${q.id} names the secret it protects`, q.secret_facts.length > 0 && q.secret_facts.some((f) => text.toLowerCase().includes(String(f).toLowerCase())), JSON.stringify(q.secret_facts));
	}
	if (q.type === 'relationship') {
		const named = q.expected_answer.match(/\b(?:DEV-2025-\d+|CC-\d+|SOP-\d+|P-\d+)\b/g) || [];
		check(`${q.id} named records appear in the evidence`, named.length > 0 && named.every((n) => text.includes(n)), named.join(', '));
	}
}

// numeric answers, recomputed here from the data files
const expectNumeric = {
	'second quarter': () => deviations.filter((d) => d.site_id === 'MB' && d.opened_date.slice(0, 7) >= '2025-04' && d.opened_date.slice(0, 7) <= '2025-06').length,
	'Thyroid Marker': () => batches.filter((b) => products.find((p) => p.id === b.product_id).name === 'Thyroid Marker Assay Kit' && b.status === 'rejected').length,
	'Elsin Park': () => { let t = 0; for (const b of batches) if (sites.find((s) => s.id === b.site_id).name === 'Elsin Park' && b.status === 'released') t += b.quantity_units; return t; },
	'major deviations': () => deviations.filter((d) => d.severity === 'major' && d.opened_date.startsWith('2025')).length,
	'status of hold': () => batches.filter((b) => b.status === 'hold').length,
	'still open': () => deviations.filter((d) => !d.closed_date).length
};
const numeric = golden.filter((q) => q.type === 'numeric');
for (const [key, fn] of Object.entries(expectNumeric)) {
	const q = numeric.find((x) => x.text.includes(key));
	check(`numeric "${key}" recomputes to the expected value`, !!q && fn() === q.expected_value, q ? `${fn()} vs ${q.expected_value}` : 'question not found');
}
check('numeric answers are non-trivial', numeric.every((q) => q.expected_value > 0));

// out-of-scope questions really are out of scope
for (const term of ['employ', 'sterilis', 'steriliz', 'regulatory inspection', 'headcount']) {
	check(`corpus has nothing on "${term}"`, !all.some((d) => d.raw.toLowerCase().includes(term)));
}

// ------------------------------------------------------------------- report
for (const r of results.filter((r) => !r.ok)) console.log(`FAIL  ${r.name}${r.detail ? `  [${r.detail}]` : ''}`);
console.log(`${results.length - failures} of ${results.length} checks passed (corpus ${manifest.corpus_version})`);
process.exit(failures ? 1 : 0);
