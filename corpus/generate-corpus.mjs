#!/usr/bin/env node
// RAG Lab corpus generator.
//
// Produces the synthetic corpus, the relational data and the golden set from a
// fixed seed, so the same command always yields the same files. Everything here
// is invented: Calder Ridge Diagnostics, its sites, products, procedures,
// batches, deviations, prices and audit findings do not describe any real
// organisation.
//
// Usage: node corpus/generate-corpus.mjs
// Output: corpus/documents/*.md, corpus/data/*.json,
//         corpus/golden.json, corpus/manifest.json

import { mkdirSync, writeFileSync, rmSync, existsSync, readdirSync } from 'node:fs';
import { createHash } from 'node:crypto';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const HERE = dirname(fileURLToPath(import.meta.url));
const OUT = HERE;
const SEED = 20261001;
const COMPANY = 'Calder Ridge Diagnostics';
const NOTICE =
	'Synthetic document. Calder Ridge Diagnostics and everything described here are invented for a retrieval evaluation demo.';

// ---------------------------------------------------------------- utilities

function mulberry32(a) {
	return function () {
		a |= 0;
		a = (a + 0x6d2b79f5) | 0;
		let t = Math.imul(a ^ (a >>> 15), 1 | a);
		t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
		return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
	};
}
const rand = mulberry32(SEED);
const pick = (arr) => arr[Math.floor(rand() * arr.length)];
const randInt = (lo, hi) => lo + Math.floor(rand() * (hi - lo + 1));
const pad = (n, w) => String(n).padStart(w, '0');
const slug = (s) =>
	s
		.toLowerCase()
		.replace(/[^a-z0-9]+/g, '-')
		.replace(/^-|-$/g, '');
const isoDate = (d) => d.toISOString().slice(0, 10);
const addDays = (iso, n) => {
	const d = new Date(iso + 'T00:00:00Z');
	d.setUTCDate(d.getUTCDate() + n);
	return isoDate(d);
};
const dayOfYear2025 = (n) => addDays('2025-01-01', n);

const BOTH = ['commercial', 'quality'];

// ------------------------------------------------------------ reference data

const sites = [
	{ id: 'TC', name: 'Tallow Creek', region: 'North' },
	{ id: 'MB', name: 'Marrow Bay', region: 'Coastal' },
	{ id: 'EP', name: 'Elsin Park', region: 'Central' }
];

const families = {
	CC: { name: 'Clinical Chemistry', site: 'TC', handlingSop: 'SOP-004', storageSop: 'SOP-008', storage: '2 to 8 °C' },
	IA: { name: 'Immunoassay', site: 'MB', handlingSop: 'SOP-005', storageSop: 'SOP-008', storage: '2 to 8 °C' },
	HM: { name: 'Hematology', site: 'EP', handlingSop: 'SOP-006', storageSop: 'SOP-010', storage: '15 to 25 °C' },
	MX: { name: 'Molecular', site: 'MB', handlingSop: 'SOP-007', storageSop: 'SOP-009', storage: 'at or below -15 °C' }
};

const productDefs = [
	['P-101', 'Lipid Panel Reagent Kit', 'CC'],
	['P-102', 'Renal Function Reagent Kit', 'CC'],
	['P-103', 'Hepatic Enzyme Reagent Kit', 'CC'],
	['P-104', 'Thyroid Marker Assay Kit', 'IA'],
	['P-105', 'Cardiac Marker Assay Kit', 'IA'],
	['P-106', 'Ferritin Assay Kit', 'IA'],
	['P-107', 'Tri-Level Hematology Control', 'HM'],
	['P-108', 'Reticulocyte Control', 'HM'],
	['P-109', 'Hematology Calibrator Set', 'HM'],
	['P-110', 'Respiratory Panel Master Mix', 'MX'],
	['P-111', 'Enteric Panel Master Mix', 'MX'],
	['P-112', 'Extraction Control Kit', 'MX']
];

const products = productDefs.map(([id, name, family], i) => ({
	id,
	name,
	family,
	family_name: families[family].name,
	site_id: families[family].site,
	shelf_life_months: [12, 18, 24, 9, 15][i % 5],
	open_vial_days: [14, 28, 30, 7, 21, 60][i % 6],
	cv_limit_pct: [3.0, 4.5, 5.0, 6.0, 2.5, 7.5][(i * 5) % 6],
	sample_volume_ul: [10, 25, 50, 100, 5, 20][(i * 7) % 6],
	calibration_interval_days: [14, 28, 30, 60, 90][(i * 3) % 5],
	range_low: [0.5, 1, 2, 5, 10][(i * 2) % 5],
	range_high: [200, 500, 800, 1000, 2500][(i * 4) % 5],
	list_price_usd: 180 + ((i * 137) % 23) * 35
}));
const productById = Object.fromEntries(products.map((p) => [p.id, p]));
const siteById = Object.fromEntries(sites.map((s) => [s.id, s]));

// -------------------------------------------------------------------- batches

const batches = [];
{
	let n = 0;
	for (const p of products) {
		const count = randInt(11, 14);
		for (let i = 0; i < count; i++) {
			n++;
			const r = rand();
			batches.push({
				id: `B-25-${pad(n, 4)}`,
				product_id: p.id,
				site_id: p.site_id,
				mfg_date: dayOfYear2025(randInt(0, 340)),
				quantity_units: randInt(4, 40) * 50,
				status: r < 0.8 ? 'released' : r < 0.92 ? 'rejected' : 'hold'
			});
		}
	}
}

// ----------------------------------------------------------------- deviations
// Nine deviations are fixed because documents and change controls refer to
// them. The rest are generated.

const fixedDeviations = [
	// key, product, category, title, linked change
	['a', 'P-102', 'documentation', 'Change impact assessment completed late', 'CC-01'],
	['b', 'P-104', 'process', 'Conjugate exposed to light beyond the permitted time', 'CC-02'],
	['c', 'P-105', 'process', 'Conjugate left on open bench during shift change', 'CC-02'],
	['d', 'P-101', 'storage', 'Cold room excursion not detected for a full shift', 'CC-03'],
	['e', 'P-110', 'release', 'Insufficient retained samples for a repeat investigation', 'CC-04'],
	['f', 'P-107', 'release', 'Retained samples exhausted during complaint testing', 'CC-04'],
	['g', 'P-108', 'equipment', 'Filling balance found out of tolerance at calibration', 'CC-05'],
	['h', 'P-103', 'supplier', 'Raw material lot received without certificate of analysis', 'CC-06'],
	['i', 'P-111', 'supplier', 'Primer lot from a supplier with a lapsed qualification', 'CC-06']
];

const deviations = [];
const devByKey = {};
{
	let n = 0;
	const nextId = () => `DEV-2025-${pad(++n, 3)}`;
	const batchFor = (pid) => batches.filter((b) => b.product_id === pid);
	// spread the fixed ones through the first half of the year
	fixedDeviations.forEach(([key, pid, category, title, change], idx) => {
		// earliest batch of the product, so the dates stay inside 2025
		const b = [...batchFor(pid)].sort((x, y) => (x.mfg_date < y.mfg_date ? -1 : 1))[0];
		const opened = addDays(b.mfg_date, 3 + idx);
		const row = {
			id: null,
			site_id: productById[pid].site_id,
			product_id: pid,
			batch_id: b.id,
			category,
			severity: 'major',
			title,
			opened_date: opened,
			closed_date: addDays(opened, 21 + idx * 2),
			linked_change_id: change
		};
		devByKey[key] = row;
		deviations.push(row);
	});
	const cats = ['documentation', 'process', 'storage', 'equipment', 'supplier', 'labeling', 'training'];
	const titles = {
		documentation: 'Batch record entry corrected without a dated signature',
		process: 'In-process check performed outside its time window',
		storage: 'Temperature logger download missed',
		equipment: 'Preventive maintenance completed after its due date',
		supplier: 'Incoming inspection sample size below plan',
		labeling: 'Label reconciliation count mismatch',
		training: 'Operator training record not current at time of task'
	};
	for (let i = 0; i < 118; i++) {
		const b = pick(batches);
		const opened = addDays(b.mfg_date, randInt(0, 20));
		if (opened > '2025-12-31') continue;
		const category = pick(cats);
		const r = rand();
		const open = rand() < 0.12;
		deviations.push({
			id: null,
			site_id: b.site_id,
			product_id: b.product_id,
			batch_id: b.id,
			category,
			severity: r < 0.7 ? 'minor' : 'major',
			title: titles[category],
			opened_date: opened,
			closed_date: open ? null : addDays(opened, randInt(5, 50)),
			linked_change_id: null
		});
	}
	deviations.sort((x, y) => (x.opened_date < y.opened_date ? -1 : x.opened_date > y.opened_date ? 1 : 0));
	for (const d of deviations) d.id = nextId();
}

// Each change control takes effect 30 days after its last source deviation
// closed. A revised procedure takes effect 14 days after its change control.
const changeSources = { 'CC-01': ['a'], 'CC-02': ['b', 'c'], 'CC-03': ['d'], 'CC-04': ['e', 'f'], 'CC-05': ['g'], 'CC-06': ['h', 'i'] };
const changeEffective = Object.fromEntries(
	Object.entries(changeSources).map(([id, keys]) => [
		id,
		addDays(keys.map((k) => devByKey[k].closed_date).sort().at(-1), 30)
	])
);

// ------------------------------------------------------------------ documents

const docs = [];
function addDoc({ id, title, type, version = '1', status = 'current', supersededBy = null, groups = BOTH, effective, sections }) {
	docs.push({ id, title, type, version, status, superseded_by: supersededBy, groups, effective, sections });
}
const sec = (heading, body) => ({ heading, slug: slug(heading), body: body.trim() });
const ref = (docId, heading) => `${docId}#${slug(heading)}`;

// ---- SOPs

const sopDefs = [
	{
		n: 1, title: 'Document Control', area: 'Quality Systems', applies: 'all sites',
		params: [['Periodic review interval', '24 months'], ['Approval signatures required', '2']],
		steps: ['Draft or revise the document in the controlled template.', 'Route for review by the process owner and Quality.', 'Record approval and set the effective date.', 'Withdraw the previous version from points of use on the effective date.'],
		refs: ['SOP-003']
	},
	{
		n: 2, title: 'Deviation Management', area: 'Quality Systems', applies: 'all sites',
		params: [['Initial assessment due', '2 business days from discovery'], ['Closure target for major deviations', '30 calendar days'], ['Closure target for minor deviations', '45 calendar days']],
		steps: ['Record the deviation on the day it is discovered.', 'Classify severity as minor or major.', 'Investigate root cause in proportion to severity.', 'Raise a change control when a procedure, material or equipment change is needed.'],
		refs: ['SOP-003', 'SOP-012']
	},
	{
		n: 3, title: 'Change Control', area: 'Quality Systems', applies: 'all sites', versioned: true, change: 'CC-01',
		params: [['Impact assessment due', ['5 business days from submission', '3 business days from submission']], ['Approvers required', '3']],
		steps: ['Submit the change request with the reason and scope.', 'Complete the impact assessment within the stated time.', 'Obtain approvals before implementation.', 'Verify effectiveness after implementation and close the record.'],
		refs: ['SOP-001', 'SOP-002']
	},
	{
		n: 4, title: 'Reagent Bulk Hold', area: 'Manufacturing', applies: 'Clinical Chemistry products',
		params: [['Maximum bulk hold time before filling', '72 hours'], ['Hold temperature', '2 to 8 °C'], ['Mixing before filling', '15 minutes']],
		steps: ['Label the bulk vessel with batch, time of formulation and hold expiry.', 'Store the bulk in the designated cold room.', 'Mix before filling for the stated time.', 'Reject bulk that exceeds the maximum hold time.'],
		refs: ['SOP-008', 'SOP-011']
	},
	{
		n: 5, title: 'Immunoassay Conjugate Handling', area: 'Manufacturing', applies: 'Immunoassay products', versioned: true, change: 'CC-02',
		params: [['Maximum cumulative light exposure', ['30 minutes', '20 minutes']], ['Maximum conjugate hold time', '48 hours'], ['Container type', 'amber, closed']],
		steps: ['Dispense conjugate under reduced lighting.', 'Record the start and end time of every light exposure.', 'Return conjugate to cold storage between operations.', 'Raise a deviation if cumulative exposure exceeds the limit.'],
		refs: ['SOP-008', 'SOP-002']
	},
	{
		n: 6, title: 'Hematology Control Filling', area: 'Manufacturing', applies: 'Hematology products',
		params: [['Fill volume tolerance', 'plus or minus 2 percent'], ['Line clearance check interval', '4 hours'], ['Cell suspension mixing speed', '60 rpm']],
		steps: ['Verify line clearance before the first fill.', 'Check fill volume at the start of each run and at the stated interval.', 'Keep the cell suspension mixing throughout filling.', 'Reconcile labels at the end of the run.'],
		refs: ['SOP-010', 'SOP-013']
	},
	{
		n: 7, title: 'Molecular Master Mix Preparation', area: 'Manufacturing', applies: 'Molecular products',
		params: [['Maximum thaw cycles for enzyme components', '3'], ['Maximum bench time during preparation', '45 minutes'], ['Preparation area', 'template-free room']],
		steps: ['Thaw components on a cold block and record the thaw count.', 'Prepare the mix in the template-free room.', 'Keep total bench time within the limit.', 'Return the mix to frozen storage immediately after dispensing.'],
		refs: ['SOP-009', 'SOP-014']
	},
	{
		n: 8, title: 'Cold Chain Storage', area: 'Warehousing', applies: 'Clinical Chemistry and Immunoassay products', versioned: true, change: 'CC-03',
		params: [['Storage range', '2 to 8 °C'], ['Cumulative excursion limit above 8 °C', ['12 hours', '6 hours']], ['Logger download interval', '24 hours']],
		steps: ['Store product only in mapped cold rooms.', 'Download temperature loggers at the stated interval.', 'Quarantine product when the cumulative excursion limit is exceeded.', 'Record every excursion as a deviation.'],
		refs: ['SOP-002', 'SOP-013']
	},
	{
		n: 9, title: 'Frozen Storage', area: 'Warehousing', applies: 'Molecular products',
		params: [['Storage temperature', 'at or below -15 °C'], ['Cumulative excursion limit above -15 °C', '2 hours'], ['Freezer alarm test interval', '3 months']],
		steps: ['Store product only in qualified freezers.', 'Respond to freezer alarms within 30 minutes.', 'Quarantine product when the cumulative excursion limit is exceeded.', 'Test freezer alarms at the stated interval.'],
		refs: ['SOP-002', 'SOP-013']
	},
	{
		n: 10, title: 'Ambient Storage', area: 'Warehousing', applies: 'Hematology products',
		params: [['Storage range', '15 to 25 °C'], ['Relative humidity limit', '60 percent'], ['Warehouse mapping interval', '36 months']],
		steps: ['Store product only in mapped ambient zones.', 'Review temperature and humidity records weekly.', 'Quarantine product stored outside the stated range.', 'Repeat warehouse mapping at the stated interval.'],
		refs: ['SOP-002']
	},
	{
		n: 11, title: 'Batch Record Review', area: 'Quality Operations', applies: 'all sites',
		params: [['Review due', '5 business days from batch completion'], ['Second-person verification', 'required for all critical steps']],
		steps: ['Confirm every entry is complete, legible and signed.', 'Confirm deviations are recorded and assessed.', 'Return incomplete records to manufacturing.', 'Forward reviewed records for batch release.'],
		refs: ['SOP-012', 'SOP-002']
	},
	{
		n: 12, title: 'Batch Release', area: 'Quality Operations', applies: 'all sites', versioned: true, change: 'CC-04',
		params: [['Retained samples per batch', ['6 units', '10 units']], ['Release decision due', '2 business days from completed review'], ['Retention period', 'shelf life plus 12 months']],
		steps: ['Confirm batch record review is complete.', 'Confirm all linked deviations are closed or assessed as not affecting release.', 'Take retained samples in the stated quantity.', 'Record the release decision.'],
		refs: ['SOP-011', 'SOP-002']
	},
	{
		n: 13, title: 'Equipment Calibration', area: 'Engineering', applies: 'all sites',
		params: [['Calibration interval for balances and pipettes', '6 months'], ['Out-of-tolerance report due', '1 business day'], ['Calibration label', 'required on every instrument']],
		steps: ['Calibrate against traceable standards.', 'Label the instrument with the calibration and due dates.', 'Report out-of-tolerance results within the stated time.', 'Assess the impact on batches made since the last passing calibration.'],
		refs: ['SOP-002']
	},
	{
		n: 14, title: 'Environmental Monitoring', area: 'Quality Operations', applies: 'all sites',
		params: [['Viable air sampling frequency', 'every 7 days'], ['Action limit for viable air samples', '100 CFU per cubic metre'], ['Template-free room surface testing', 'every 14 days']],
		steps: ['Sample at the mapped locations.', 'Incubate and read plates per the method.', 'Investigate any result above the action limit.', 'Trend results quarterly.'],
		refs: ['SOP-002']
	},
	{
		n: 15, title: 'Supplier Qualification', area: 'Supply Chain', applies: 'all sites',
		params: [['Requalification interval', '36 months'], ['On-site audit', 'required for critical suppliers'], ['Certificate of analysis', 'required with every lot']],
		steps: ['Classify the supplier as critical or non-critical.', 'Qualify the supplier before first purchase.', 'Verify a certificate of analysis with every lot received.', 'Requalify at the stated interval.'],
		refs: ['SOP-003']
	},
	{
		n: 16, title: 'Complaint Handling', area: 'Customer Quality', applies: 'all sites',
		params: [['Acknowledgement due', '3 business days'], ['Investigation due', '30 calendar days'], ['Retained sample testing', 'performed when the complaint concerns product performance']],
		steps: ['Log the complaint on the day it is received.', 'Acknowledge within the stated time.', 'Test retained samples when relevant.', 'Close the complaint with the investigation result.'],
		refs: ['SOP-012', 'SOP-002']
	}
];

const sopId = (n) => `SOP-${pad(n, 3)}`;
const sopParamValue = (def, label, version) => {
	const p = def.params.find((x) => x[0] === label);
	return Array.isArray(p[1]) ? p[1][version - 1] : p[1];
};
const sopDocId = (def, version) => (def.versioned ? `${sopId(def.n)}-v${version}` : sopId(def.n));
const currentSopDocId = (n) => {
	const def = sopDefs.find((d) => d.n === n);
	return sopDocId(def, def.versioned ? 2 : 1);
};

for (const def of sopDefs) {
	const versions = def.versioned ? [1, 2] : [1];
	for (const v of versions) {
		const id = sopDocId(def, v);
		const current = v === versions.length;
		const effective = v === 1 ? `2024-${pad(((def.n * 5) % 12) + 1, 2)}-01` : addDays(changeEffective[def.change], 14);
		const paramLines = def.params
			.map(([label, value]) => `- ${label}: ${Array.isArray(value) ? value[v - 1] : value}`)
			.join('\n');
		const history =
			v === 1
				? 'Version 1. Initial issue.'
				: `Version 2. Revised under change control ${def.change}. Supersedes version 1, which must not be used.`;
		addDoc({
			id,
			title: `${sopId(def.n)} ${def.title}${def.versioned ? ` (version ${v})` : ''}`,
			type: 'sop',
			version: String(v),
			status: current ? 'current' : 'superseded',
			supersededBy: current ? null : sopDocId(def, 2),
			effective,
			sections: [
				sec('Purpose', `${NOTICE}\n\nThis procedure defines how ${COMPANY} performs ${def.title.toLowerCase()}. It belongs to the ${def.area} area. Document number ${sopId(def.n)}, version ${v}, status ${current ? 'current' : 'superseded'}.`),
				sec('Scope', `${sopId(def.n)} applies to ${def.applies}. It covers routine operation and does not cover validation studies.`),
				sec('Definitions', `Deviation: an unplanned departure from an approved procedure. Change control: the documented review and approval of a change before it is made. Process owner: the person accountable for ${def.title.toLowerCase()}.`),
				sec('Responsibilities', `The process owner maintains ${sopId(def.n)}. Operators follow it and record their work. Quality reviews the records and assesses any deviation under SOP-002.`),
				sec('Parameters', `The controlled parameters for ${sopId(def.n)} ${def.title}, version ${v}, are:\n\n${paramLines}`),
				sec('Procedure', def.steps.map((s, i) => `${i + 1}. ${s}`).join('\n')),
				sec('Records', `Records generated under ${sopId(def.n)} are kept with the batch record or the quality system file and are subject to SOP-001 Document Control.`),
				sec('References and history', `Related procedures: ${def.refs.join(', ')}.\n\n${history} Effective ${effective}.`)
			]
		});
	}
}

// ---- product specification sheets

for (const p of products) {
	const fam = families[p.family];
	const id = `SPEC-${p.id}`;
	addDoc({
		id,
		title: `Product specification: ${p.id} ${p.name}`,
		type: 'spec',
		effective: '2024-11-15',
		sections: [
			sec('Overview', `${NOTICE}\n\n${p.id} ${p.name} is a ${fam.name} product manufactured at the ${siteById[p.site_id].name} site (${p.site_id}). Handling during manufacture follows ${fam.handlingSop}. Storage of finished product follows ${fam.storageSop}.`),
			sec('Intended use', `${p.name} is intended for in vitro diagnostic use by trained laboratory staff on compatible analysers. It is not intended for self-testing.`),
			sec('Specifications', `Specification table for ${p.id} ${p.name}.\n\n| Parameter | Limit |\n| --- | --- |\n| Within-run precision, upper limit (CV %) | ${p.cv_limit_pct.toFixed(1)} |\n| Measuring range | ${p.range_low} to ${p.range_high} units |\n| Sample volume (µL) | ${p.sample_volume_ul} |\n| Calibration interval (days) | ${p.calibration_interval_days} |\n| Shelf life (months) | ${p.shelf_life_months} |\n| Open-vial stability (days) | ${p.open_vial_days} |`),
			sec('Storage and handling', `Store ${p.id} ${fam.storage}. Do not use product that has been stored outside this range for longer than the excursion limit in ${fam.storageSop}.`),
			sec('Revision history', `Revision 1, effective 2024-11-15. Owner: Product Quality, ${siteById[p.site_id].name}.`)
		]
	});
}

// ---- deviation reports (one per fixed deviation)

const drIdByKey = {};
fixedDeviations.forEach(([key, pid, category, title, change], idx) => {
	const d = devByKey[key];
	const p = productById[pid];
	const id = `DR-${pad(idx + 1, 2)}`;
	drIdByKey[key] = id;
	const causes = {
		documentation: 'the assessment task had no named owner once the request was submitted',
		process: 'the procedure allowed more exposure time than the material tolerates and exposure was not timed',
		storage: 'the excursion limit was long enough that a single missed logger download hid the event',
		release: 'the number of retained samples was too small to support more than one investigation',
		equipment: 'the balance drifted between calibrations and no interim check was scheduled',
		supplier: 'the receiving check did not block material from a supplier with incomplete documentation'
	};
	addDoc({
		id,
		title: `Deviation report ${id}: ${title}`,
		type: 'deviation_report',
		effective: d.closed_date,
		sections: [
			sec('Summary', `${NOTICE}\n\nDeviation report ${id} covers deviation ${d.id}, a major ${category} deviation at the ${siteById[d.site_id].name} site. ${title}. It affected batch ${d.batch_id} of ${p.id} ${p.name}.`),
			sec('Description', `On ${d.opened_date}, staff at ${siteById[d.site_id].name} recorded deviation ${d.id}: ${title.toLowerCase()}. The batch concerned was ${d.batch_id}. The deviation was classified as major under SOP-002.`),
			sec('Investigation', `The investigation for ${d.id} reviewed the batch record, interviewed the staff involved and checked whether other batches of ${p.id} were affected. No other batch was found to be affected.`),
			sec('Root cause', `The root cause of ${d.id} was that ${causes[category]}.`),
			sec('Corrective actions', `Batch ${d.batch_id} was assessed under SOP-012 before any release decision. A change was requested so the same cause cannot recur. The deviation was closed on ${d.closed_date}.`),
			sec('Links', `Deviation ${d.id} led to change control ${change}. Product: ${p.id}. Site: ${d.site_id}. Batch: ${d.batch_id}.`)
		]
	});
});

// ---- change controls

const changeDefs = [
	{ id: 'CC-01', keys: changeSources['CC-01'], sop: 3, what: 'Shortened the impact assessment time in SOP-003 from 5 to 3 business days and named an owner for each assessment.' },
	{ id: 'CC-02', keys: changeSources['CC-02'], sop: 5, what: 'Reduced the maximum cumulative light exposure in SOP-005 from 30 to 20 minutes and required exposure to be timed.' },
	{ id: 'CC-03', keys: changeSources['CC-03'], sop: 8, what: 'Reduced the cumulative excursion limit in SOP-008 from 12 to 6 hours.' },
	{ id: 'CC-04', keys: changeSources['CC-04'], sop: 12, what: 'Increased retained samples per batch in SOP-012 from 6 to 10 units.' },
	{ id: 'CC-05', keys: changeSources['CC-05'], sop: null, relatedSop: 13, what: 'Added a monthly interim check for filling balances at Elsin Park. No procedure was revised; the calibration schedule was updated under SOP-013.' },
	{ id: 'CC-06', keys: changeSources['CC-06'], sop: null, relatedSop: 15, what: 'Added a receiving block for lots from suppliers with incomplete or lapsed qualification. No procedure was revised; the receiving checklist was updated under SOP-015.' }
];

for (const c of changeDefs) {
	const devs = c.keys.map((k) => devByKey[k]);
	const devIds = devs.map((d) => d.id).join(' and ');
	const revised = c.sop ? `${sopId(c.sop)} was revised to version 2 (${sopDocId(sopDefs.find((s) => s.n === c.sop), 2)}).` : `No SOP was revised. Related procedure: ${sopId(c.relatedSop)}.`;
	addDoc({
		id: c.id,
		title: `Change control ${c.id}`,
		type: 'change_control',
		effective: changeEffective[c.id],
		sections: [
			sec('Summary', `${NOTICE}\n\nChange control ${c.id} was raised in response to ${devs.length === 1 ? 'deviation' : 'deviations'} ${devIds}. ${c.what}`),
			sec('Reason for change', `${devs.map((d) => `${d.id} (${d.title.toLowerCase()}, ${siteById[d.site_id].name})`).join('; ')}. The investigation concluded that the existing control was not sufficient.`),
			sec('Change description', c.what),
			sec('Impact assessment', `The change was assessed for impact on product quality, validation status, training and documents. Training was required for all staff working under the affected procedure. No revalidation was required.`),
			sec('Approvals', `Approved by the process owner, Quality and the site head, as required by SOP-003.`),
			sec('Links', `Source deviations: ${devs.map((d) => d.id).join(', ')}. ${revised} Products involved: ${[...new Set(devs.map((d) => d.product_id))].join(', ')}.`)
		]
	});
}

// ---- commercial memos (Commercial group only)

const COMMERCIAL = ['commercial'];
const priceTable = (ps) =>
	`| Product | List price per kit (USD) |\n| --- | --- |\n${ps.map((p) => `| ${p.id} ${p.name} | ${p.list_price_usd} |`).join('\n')}`;
const claims = (ps) =>
	ps
		.map((p) => `${p.id} ${p.name}: storage ${families[p.family].storage}, shelf life ${p.shelf_life_months} months, open-vial stability ${p.open_vial_days} days.`)
		.join(' ');
const famProducts = (f) => products.filter((p) => p.family === f);

const memoDefs = [
	{ title: 'Clinical Chemistry price list 2026', body: priceTable(famProducts('CC')), ps: famProducts('CC') },
	{ title: 'Immunoassay price list 2026', body: priceTable(famProducts('IA')), ps: famProducts('IA') },
	{ title: 'Hematology price list 2026', body: priceTable(famProducts('HM')), ps: famProducts('HM') },
	{ title: 'Molecular price list 2026', body: priceTable(famProducts('MX')), ps: famProducts('MX') },
	{ title: 'Distributor discount policy', body: 'Distributor contracts may carry a discount of up to 18 percent off list price. Discounts above 12 percent need approval from the commercial director. Tender pricing is handled case by case.', ps: famProducts('CC') },
	{ title: 'Molecular panel launch plan', body: 'P-110 Respiratory Panel Master Mix launches to existing accounts in the first quarter of 2026 and to new accounts in the second quarter. Launch stock target: 4,000 kits.', ps: famProducts('MX') },
	{ title: 'Cold chain messaging for sales teams', body: 'Sales teams may state that refrigerated products are stored at 2 to 8 °C and that temperature excursions are controlled by procedure. Do not quote the excursion limit to customers; refer technical questions to Quality.', ps: [...famProducts('CC'), ...famProducts('IA')] },
	{ title: 'Hematology controls account review', body: 'Three key accounts renew their hematology control contracts in 2026. Renewal pricing holds list price with a loyalty discount of 8 percent.', ps: famProducts('HM') }
];
memoDefs.forEach((m, i) => {
	const id = `MEMO-${pad(i + 1, 2)}`;
	addDoc({
		id,
		title: `Commercial memo ${id}: ${m.title}`,
		type: 'commercial_memo',
		groups: COMMERCIAL,
		effective: `2025-${pad(9 + (i % 3), 2)}-10`,
		sections: [
			sec('Summary', `${NOTICE}\n\nCommercial memo ${id}, ${m.title}. Restricted to the Commercial group. Not for distribution outside Commercial.`),
			sec('Details', m.body),
			sec('Product claims', `Approved product statements for this memo. ${claims(m.ps)}`),
			sec('Next steps', `Account managers apply this memo from its effective date. Questions go to the commercial director.`)
		]
	});
});

// ---- quality audit findings (Quality group only)

const QUALITY = ['quality'];
const auditDefs = [
	{ sop: 8, cls: 'major', finding: 'Logger downloads at the Tallow Creek cold room were missed on four days in one month.', req: 'SOP-008 requires temperature loggers to be downloaded every 24 hours and limits cumulative excursions above 8 °C.' },
	{ sop: 13, cls: 'minor', finding: 'Two pipettes at Marrow Bay carried calibration labels without a due date.', req: 'SOP-013 requires a calibration label on every instrument and a calibration interval of 6 months for balances and pipettes.' },
	{ sop: 12, cls: 'critical', finding: 'One batch at Elsin Park was released while a linked major deviation was still open and unassessed.', req: 'SOP-012 requires all linked deviations to be closed or assessed before release.' },
	{ sop: 15, cls: 'major', finding: 'A critical supplier was last audited on site more than 36 months ago.', req: 'SOP-015 requires requalification every 36 months and an on-site audit for critical suppliers.' },
	{ sop: 4, cls: 'minor', finding: 'A bulk vessel label at Tallow Creek did not show the hold expiry time.', req: 'SOP-004 requires the vessel label to show the hold expiry and limits bulk hold time to 72 hours.' }
];
auditDefs.forEach((a, i) => {
	const id = `AF-${pad(i + 1, 2)}`;
	addDoc({
		id,
		title: `Internal audit finding ${id}`,
		type: 'audit_finding',
		groups: QUALITY,
		effective: `2025-${pad(5 + i, 2)}-20`,
		sections: [
			sec('Scope', `${NOTICE}\n\nInternal audit finding ${id}. Restricted to the Quality group. The audit covered compliance with ${sopId(a.sop)}.`),
			sec('Finding', `${a.finding} Classification: ${a.cls}.`),
			sec('Requirement', a.req),
			sec('Evidence', `Evidence was taken from records sampled during the audit and from interviews with the staff responsible.`),
			sec('Response', `The area owner accepted finding ${id} and committed to corrective action within ${a.cls === 'critical' ? 15 : a.cls === 'major' ? 30 : 60} calendar days.`)
		]
	});
});

// ----------------------------------------------------------------- golden set

const golden = [];
let qn = 0;
function q({ type, text, group, answer, value = null, verdict = 'answered', refs = [], hypothesis, facts = [], secrets = [] }) {
	golden.push({
		id: `Q-${pad(++qn, 2)}`,
		type,
		text,
		ask_as_group: group,
		expected_verdict: verdict,
		expected_answer: answer,
		expected_value: value,
		// key_facts: every entry must appear in a correct answer; an array entry means any one of its members.
		key_facts: facts,
		// secret_facts: restricted values that must never appear in an answer to this question.
		secret_facts: secrets,
		supporting_refs: refs,
		hypothesis
	});
}
const S = (docId, heading) => ({ kind: 'section', ref: ref(docId, heading) });
const R = (table, where) => ({ kind: 'rows', table, where });
const sopDef = (n) => sopDefs.find((d) => d.n === n);
const P = (n, label) => sopParamValue(sopDef(n), label, sopDef(n).versioned ? 2 : 1);
// The figure a correct answer must contain: the first number and the word after it ("72 hours").
const F = (n, label) => {
	const m = P(n, label).match(/(\d+(?:\.\d+)?)(?:\s+([A-Za-z]+))?/);
	return m[2] ? `${m[1]} ${m[2]}` : m[1];
};

// lookup (8): single parameter from a non-versioned procedure
q({ type: 'lookup', group: 'quality', hypothesis: 'hybrid', text: 'How long can clinical chemistry reagent bulk be held before it has to be filled?', answer: `${P(4, 'Maximum bulk hold time before filling')} (SOP-004).`, facts: [F(4, 'Maximum bulk hold time before filling')], refs: [S('SOP-004', 'Parameters')] });
q({ type: 'lookup', group: 'commercial', hypothesis: 'hybrid', text: 'What does SOP-013 give as the calibration interval for balances and pipettes?', answer: `${P(13, 'Calibration interval for balances and pipettes')}.`, facts: [F(13, 'Calibration interval for balances and pipettes')], refs: [S('SOP-013', 'Parameters')] });
q({ type: 'lookup', group: 'quality', hypothesis: 'hybrid', text: 'How many times may enzyme components be thawed when preparing a molecular master mix?', answer: `${P(7, 'Maximum thaw cycles for enzyme components')} thaw cycles (SOP-007).`, facts: [F(7, 'Maximum thaw cycles for enzyme components')], refs: [S('SOP-007', 'Parameters')] });
q({ type: 'lookup', group: 'quality', hypothesis: 'hybrid', text: 'How soon must a deviation receive its initial assessment?', answer: `Within ${P(2, 'Initial assessment due')} (SOP-002).`, facts: [F(2, 'Initial assessment due')], refs: [S('SOP-002', 'Parameters')] });
q({ type: 'lookup', group: 'commercial', hypothesis: 'hybrid', text: 'What humidity limit applies to ambient storage of hematology products?', answer: `Relative humidity of ${P(10, 'Relative humidity limit')} (SOP-010).`, facts: [F(10, 'Relative humidity limit')], refs: [S('SOP-010', 'Parameters')] });
q({ type: 'lookup', group: 'quality', hypothesis: 'hybrid', text: 'How often are suppliers requalified?', answer: `Every ${P(15, 'Requalification interval')} (SOP-015).`, facts: [F(15, 'Requalification interval')], refs: [S('SOP-015', 'Parameters')] });
q({ type: 'lookup', group: 'commercial', hypothesis: 'hybrid', text: 'Within how many days must a customer complaint be acknowledged under SOP-016?', answer: `${P(16, 'Acknowledgement due')}.`, facts: [F(16, 'Acknowledgement due')], refs: [S('SOP-016', 'Parameters')] });
q({ type: 'lookup', group: 'quality', hypothesis: 'hybrid', text: 'What is the action limit for viable air samples in environmental monitoring?', answer: `${P(14, 'Action limit for viable air samples')} (SOP-014).`, facts: [F(14, 'Action limit for viable air samples')], refs: [S('SOP-014', 'Parameters')] });

// multi-hop (6): product or record -> procedure -> parameter
{
	const p101 = productById['P-101'], p110 = productById['P-110'], p111 = productById['P-111'], p108 = productById['P-108'];
	q({ type: 'multi_hop', group: 'quality', hypothesis: 'agentic', text: `What cumulative temperature excursion limit applies to storing ${p101.name}?`, answer: `${P(8, 'Cumulative excursion limit above 8 °C')} above 8 °C. ${p101.id} is stored under SOP-008, current version 2.`, facts: [F(8, 'Cumulative excursion limit above 8 °C')], refs: [S('SPEC-P-101', 'Overview'), S('SOP-008-v2', 'Parameters')] });
	q({ type: 'multi_hop', group: 'commercial', hypothesis: 'agentic', text: `What cumulative temperature excursion limit applies to storing ${p110.name}?`, answer: `${P(9, 'Cumulative excursion limit above -15 °C')} above -15 °C. ${p110.id} is stored under SOP-009.`, facts: [F(9, 'Cumulative excursion limit above -15 °C')], refs: [S('SPEC-P-110', 'Overview'), S('SOP-009', 'Parameters')] });
	q({ type: 'multi_hop', group: 'quality', hypothesis: 'agentic', text: `What is the maximum bench time allowed during manufacture of ${p111.name}?`, answer: `${P(7, 'Maximum bench time during preparation')}. ${p111.id} is handled under SOP-007.`, facts: [F(7, 'Maximum bench time during preparation')], refs: [S('SPEC-P-111', 'Overview'), S('SOP-007', 'Parameters')] });
	q({ type: 'multi_hop', group: 'quality', hypothesis: 'agentic', text: `Which site makes ${p108.name}, and how often is line clearance checked under the handling procedure for that product?`, answer: `${siteById[p108.site_id].name}. Line clearance is checked every ${P(6, 'Line clearance check interval')} under SOP-006.`, facts: [siteById[p108.site_id].name, F(6, 'Line clearance check interval')], refs: [S('SPEC-P-108', 'Overview'), S('SOP-006', 'Parameters')] });
	const dg = devByKey['g'];
	q({ type: 'multi_hop', group: 'commercial', hypothesis: 'agentic', text: `Deviation ${dg.id} concerned a batch of which product, and what humidity limit applies to storing that product?`, answer: `${dg.product_id} ${productById[dg.product_id].name}. Storage follows SOP-010, with a relative humidity limit of ${P(10, 'Relative humidity limit')}.`, facts: [[dg.product_id, productById[dg.product_id].name], F(10, 'Relative humidity limit')], refs: [S(drIdByKey['g'], 'Summary'), S(`SPEC-${dg.product_id}`, 'Overview'), S('SOP-010', 'Parameters')] });
	q({ type: 'multi_hop', group: 'quality', hypothesis: 'agentic', text: 'Change control CC-04 revised a procedure. What is the current value of the parameter it changed?', answer: `Retained samples per batch is now ${P(12, 'Retained samples per batch')} (SOP-012 version 2).`, facts: [F(12, 'Retained samples per batch')], refs: [S('CC-04', 'Change description'), S('SOP-012-v2', 'Parameters')] });
}

// numeric (6): answered from the relational tables
{
	const q2 = deviations.filter((d) => d.site_id === 'MB' && d.opened_date >= '2025-04-01' && d.opened_date <= '2025-06-30').length;
	q({ type: 'numeric', group: 'quality', hypothesis: 'sql', text: 'How many deviations did the Marrow Bay site open in the second quarter of 2025?', answer: String(q2), value: q2, facts: [String(q2)], refs: [R('deviation', "site_id = 'MB' and opened_date between 2025-04-01 and 2025-06-30")] });
	const rej = batches.filter((b) => b.product_id === 'P-104' && b.status === 'rejected').length;
	q({ type: 'numeric', group: 'quality', hypothesis: 'sql', text: 'How many batches of the Thyroid Marker Assay Kit were rejected in 2025?', answer: String(rej), value: rej, facts: [String(rej)], refs: [R('batch', "product_id = 'P-104' and status = 'rejected'")] });
	const qty = batches.filter((b) => b.site_id === 'EP' && b.status === 'released').reduce((s, b) => s + b.quantity_units, 0);
	q({ type: 'numeric', group: 'commercial', hypothesis: 'sql', text: 'What is the total quantity, in units, of released batches made at Elsin Park?', answer: String(qty), value: qty, facts: [String(qty)], refs: [R('batch', "site_id = 'EP' and status = 'released'")] });
	const maj = deviations.filter((d) => d.severity === 'major').length;
	q({ type: 'numeric', group: 'quality', hypothesis: 'sql', text: 'How many major deviations were opened in 2025 across all sites?', answer: String(maj), value: maj, facts: [String(maj)], refs: [R('deviation', "severity = 'major'")] });
	const hold = batches.filter((b) => b.status === 'hold').length;
	q({ type: 'numeric', group: 'commercial', hypothesis: 'sql', text: 'How many batches currently have a status of hold?', answer: String(hold), value: hold, facts: [String(hold)], refs: [R('batch', "status = 'hold'")] });
	const open = deviations.filter((d) => d.closed_date === null).length;
	q({ type: 'numeric', group: 'quality', hypothesis: 'sql', text: 'How many deviations are still open, with no closed date?', answer: String(open), value: open, facts: [String(open)], refs: [R('deviation', 'closed_date is null')] });
}

// cross-document relationship (6)
{
	const d = (k) => devByKey[k];
	q({ type: 'relationship', group: 'quality', hypothesis: 'graph', text: `Which change control was raised because of deviation ${d('d').id}?`, answer: 'CC-03.', facts: ['CC-03'], refs: [S(drIdByKey['d'], 'Links'), S('CC-03', 'Links')] });
	q({ type: 'relationship', group: 'quality', hypothesis: 'graph', text: 'Which deviations led to change control CC-02?', answer: `${d('b').id} and ${d('c').id}.`, facts: [d('b').id, d('c').id], refs: [S('CC-02', 'Links')] });
	q({ type: 'relationship', group: 'commercial', hypothesis: 'graph', text: `Which procedure was revised as a result of deviation ${d('e').id}?`, answer: 'SOP-012 Batch Release, revised to version 2 under CC-04.', facts: ['SOP-012'], refs: [S(drIdByKey['e'], 'Links'), S('CC-04', 'Links')] });
	const mbChanges = [...new Set(fixedDeviations.filter(([k]) => d(k).site_id === 'MB').map((f) => f[4]))].sort();
	q({ type: 'relationship', group: 'quality', hypothesis: 'graph', text: 'Which change controls were raised from deviations at the Marrow Bay site?', answer: `${mbChanges.join(', ')}.`, facts: mbChanges, refs: mbChanges.flatMap((c) => [S(c, 'Summary'), S(c, 'Reason for change')]) });
	q({ type: 'relationship', group: 'commercial', hypothesis: 'graph', text: 'Which products were involved in the deviations behind change control CC-06?', answer: `${d('h').product_id} ${productById[d('h').product_id].name} and ${d('i').product_id} ${productById[d('i').product_id].name}.`, facts: [[d('h').product_id, productById[d('h').product_id].name], [d('i').product_id, productById[d('i').product_id].name]], refs: [S('CC-06', 'Links')] });
	q({ type: 'relationship', group: 'quality', hypothesis: 'graph', text: 'Which deviations ultimately led to the current version of SOP-005?', answer: `${d('b').id} and ${d('c').id}, through change control CC-02.`, facts: [d('b').id, d('c').id], refs: [S('SOP-005-v2', 'References and history'), S('CC-02', 'Links')] });
}

// table (6): a value from a specification table
{
	const t = (pid, label, value, unit) => {
		const p = productById[pid];
		q({ type: 'table', group: pid < 'P-107' ? 'quality' : 'commercial', hypothesis: 'multimodal', text: `In the specification for ${p.name}, what is the ${label}?`, answer: `${value}${unit}.`, facts: String(value).split(' to '), refs: [S(`SPEC-${pid}`, 'Specifications')] });
	};
	t('P-102', 'upper limit for within-run precision', productById['P-102'].cv_limit_pct.toFixed(1), ' % CV');
	t('P-105', 'calibration interval', productById['P-105'].calibration_interval_days, ' days');
	t('P-106', 'sample volume', productById['P-106'].sample_volume_ul, ' µL');
	t('P-109', 'open-vial stability', productById['P-109'].open_vial_days, ' days');
	t('P-110', 'shelf life', productById['P-110'].shelf_life_months, ' months');
	t('P-112', 'measuring range', `${productById['P-112'].range_low} to ${productById['P-112'].range_high}`, ' units');
}

// superseded version (4): the current value where two versions exist
q({ type: 'superseded', group: 'quality', hypothesis: 'hybrid', text: 'How long does a change control impact assessment have to be completed in?', answer: `${P(3, 'Impact assessment due')} (SOP-003 version 2; version 1 allowed 5 business days and is superseded).`, facts: [F(3, 'Impact assessment due')], refs: [S('SOP-003-v2', 'Parameters')] });
q({ type: 'superseded', group: 'quality', hypothesis: 'hybrid', text: 'What is the maximum cumulative light exposure for immunoassay conjugate?', answer: `${P(5, 'Maximum cumulative light exposure')} (SOP-005 version 2; version 1 allowed 30 minutes and is superseded).`, facts: [F(5, 'Maximum cumulative light exposure')], refs: [S('SOP-005-v2', 'Parameters')] });
q({ type: 'superseded', group: 'commercial', hypothesis: 'hybrid', text: 'How long may refrigerated product spend above 8 °C in total before it is quarantined?', answer: `${P(8, 'Cumulative excursion limit above 8 °C')} (SOP-008 version 2; version 1 allowed 12 hours and is superseded).`, facts: [F(8, 'Cumulative excursion limit above 8 °C')], refs: [S('SOP-008-v2', 'Parameters')] });
q({ type: 'superseded', group: 'quality', hypothesis: 'hybrid', text: 'How many retained samples are taken from each batch at release?', answer: `${P(12, 'Retained samples per batch')} (SOP-012 version 2; version 1 required 6 units and is superseded).`, facts: [F(12, 'Retained samples per batch')], refs: [S('SOP-012-v2', 'Parameters')] });

// out of scope (3): the corpus has no answer
q({ type: 'out_of_scope', group: 'quality', verdict: 'insufficient_evidence', hypothesis: 'all', text: 'How many people does Calder Ridge Diagnostics employ?', answer: 'The corpus does not say.' });
q({ type: 'out_of_scope', group: 'commercial', verdict: 'insufficient_evidence', hypothesis: 'all', text: 'Which sterilisation method is used for the hematology control vials?', answer: 'The corpus does not say.' });
q({ type: 'out_of_scope', group: 'quality', verdict: 'insufficient_evidence', hypothesis: 'all', text: 'What was the result of the most recent external regulatory inspection at Marrow Bay?', answer: 'The corpus does not say.' });

// forbidden (3): the answer exists but not for the asking group
q({ type: 'forbidden', group: 'quality', verdict: 'access_denied', hypothesis: 'all', text: 'What is the list price of the Hepatic Enzyme Reagent Kit?', answer: 'Not available to the Quality group.', secrets: [String(productById['P-103'].list_price_usd)], refs: [S('MEMO-01', 'Details')] });
q({ type: 'forbidden', group: 'quality', verdict: 'access_denied', hypothesis: 'all', text: 'What is the maximum discount allowed on distributor contracts?', answer: 'Not available to the Quality group.', secrets: ['18 percent', '18%'], refs: [S('MEMO-05', 'Details')] });
q({ type: 'forbidden', group: 'commercial', verdict: 'access_denied', hypothesis: 'all', text: 'How was internal audit finding AF-03 classified?', answer: 'Not available to the Commercial group.', secrets: ['Classification: critical', 'classified as critical', 'was critical'], refs: [S('AF-03', 'Finding')] });

// --------------------------------------------------------------------- output

if (existsSync(join(OUT, 'documents'))) {
	// overwrite in place; files are regenerated with identical names
	for (const f of readdirSync(join(OUT, 'documents'))) {
		try { rmSync(join(OUT, 'documents', f)); } catch { /* deletion may be disallowed; the file is overwritten below */ }
	}
}
mkdirSync(join(OUT, 'documents'), { recursive: true });
mkdirSync(join(OUT, 'data'), { recursive: true });

const hash = createHash('sha256');
function write(rel, content) {
	writeFileSync(join(OUT, rel), content);
	hash.update(rel).update(content);
}

let sectionCount = 0;
for (const d of docs) {
	sectionCount += d.sections.length;
	const front = [
		'---',
		`id: ${d.id}`,
		`title: "${d.title}"`,
		`type: ${d.type}`,
		`version: ${d.version}`,
		`status: ${d.status}`,
		`superseded_by: ${d.superseded_by ?? ''}`,
		`groups: [${d.groups.join(', ')}]`,
		`effective: ${d.effective}`,
		'synthetic: true',
		'---',
		''
	].join('\n');
	const body = `# ${d.title}\n\n` + d.sections.map((s) => `## ${s.heading}\n\n${s.body}\n`).join('\n');
	write(`documents/${d.id}.md`, front + body);
}
write('data/site.json', JSON.stringify(sites, null, '\t') + '\n');
write('data/product.json', JSON.stringify(products, null, '\t') + '\n');
write('data/batch.json', JSON.stringify(batches, null, '\t') + '\n');
write('data/deviation.json', JSON.stringify(deviations, null, '\t') + '\n');
write('golden.json', JSON.stringify(golden, null, '\t') + '\n');

const manifest = {
	corpus_version: hash.digest('hex').slice(0, 12),
	seed: SEED,
	synthetic: true,
	notice: NOTICE,
	documents: docs.length,
	sections: sectionCount,
	rows: { site: sites.length, product: products.length, batch: batches.length, deviation: deviations.length },
	questions: golden.length
};
writeFileSync(join(OUT, 'manifest.json'), JSON.stringify(manifest, null, '\t') + '\n');
console.log(JSON.stringify(manifest, null, 2));
