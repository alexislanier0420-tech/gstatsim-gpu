// Builds the 48 x 36 in landscape poster (figures/poster/poster.pptx) from the
// figure PNGs and the large-region speed records (docs/hpg/region_ase_500m_*.json, the
// HiPerGator runs of benchmarks/bench_region.py). Run from the repo root:
//   node figures/poster/build_poster.js
// (needs pptxgenjs: npm install pptxgenjs, or NODE_PATH pointing at it)
const fs = require("fs");
const path = require("path");
const pptxgen = require("pptxgenjs");

const HERE = __dirname;
const ROOT = path.resolve(HERE, "..", "..");
const img = (name) => path.join(HERE, name);
// Headline numbers: the large region (Amundsen Sea Embayment) at 500 m, 1,153,419 cells
const ase = Object.assign({}, ...["region_ase_500m_gpu_49850616.json", "region_ase_500m_cpu_49850618.json",
  "region_ase_500m_cpu_49873080.json"].map((f) =>
  JSON.parse(fs.readFileSync(path.join(ROOT, "docs", "hpg", f), "utf8")).steps));
const xKrig = Math.round(ase.gstatsim_kriging.seconds / ase.gpu_kriging.seconds);
const xSgsMany = Math.round(ase.gstatsim_sgs_x1.seconds / ase.gpu_sgs_x32.seconds_per_realization);

// Colors (match the figures): glacier navy, GPU blue, CPU orange, inks
const NAVY = "16324F", BLUE = "2A78D6", ORANGE = "EB6834", INK = "0B0B0B", INK2 = "52514E", MUTED = "898781";
const TINT = "F3F2EF";
const FONT = "Segoe UI";

const pres = new pptxgen();
// The class asks for ~48 x 36 in landscape. The layout below is written in a
// 36 x 27 frame and scaled by K = 4/3 (positions, sizes, fonts, line widths).
const K = 4 / 3;
pres.defineLayout({ name: "POSTER_48x36", width: 48, height: 36 });
pres.layout = "POSTER_48x36";
pres.title = "GPU-accelerated geostatistics for Antarctic bed topography";
pres.author = "Alexis McGill";
pres.theme = { headFontFace: FONT, bodyFontFace: FONT };
const slide = pres.addSlide();
slide.background = { color: "FFFFFF" };
function scaled(o) {
  const r = { ...o };
  for (const key of ["x", "y", "w", "h"]) if (typeof r[key] === "number") r[key] *= K;
  for (const key of ["fontSize", "paraSpaceAfter"]) if (r[key]) r[key] = Math.round(r[key] * K);
  if (r.line && r.line.width) r.line = { ...r.line, width: r.line.width * K };
  return r;
}
const s = {
  addText: (t, o) => slide.addText(t, scaled(o)),
  addImage: (o) => slide.addImage(scaled(o)),
  addShape: (shape, o) => slide.addShape(shape, scaled(o)),
};

const M = 0.75, GUT = 0.5, COL = 8.25;
const colX = (i) => M + i * (COL + GUT);
const text = (t, o) => s.addText(t, { isTextBox: true, fontFace: FONT, margin: 0, ...o });

// ------------------------------------------------------------------ header
text("Antarctic bed maps, 100× faster on a GPU, with the same answers", {
  x: M, y: 0.55, w: 30.5, h: 1.45, fontSize: 66, bold: true, color: NAVY, valign: "top" });
text([
  { text: "Alexis McGill", options: { bold: true, color: INK } },
  { text: "   ·   GLY4930, University of Florida   ·   Gator Glaciology Lab", options: { color: INK2 } },
], { x: M, y: 2.15, w: 29.5, h: 0.6, fontSize: 32, valign: "top" });
text("A CUDA/CuPy backend for GStatSim that reproduces its kriging and sequential Gaussian simulation exactly.",
  { x: M, y: 2.85, w: 29.5, h: 0.55, fontSize: 26, italic: true, color: INK2, valign: "top" });
s.addImage({ path: img("gator_glaciology_logo.jpg"), x: 32.35, y: 0.55, w: 2.9, h: 2.9, altText: "Gator Glaciology Lab logo" });
s.addShape(pres.shapes.LINE, { x: M, y: 3.75, w: 36 - 2 * M, h: 0, line: { color: "E1E0D9", width: 1.5 } });

// ------------------------------------------------------------------ columns
function heading(i, num, title, y) {
  const x = colX(i);
  s.addShape(pres.shapes.OVAL, { x, y, w: 0.72, h: 0.72, fill: { color: NAVY }, line: { color: NAVY } });
  text(String(num), { x, y, w: 0.72, h: 0.72, fontSize: 30, bold: true, color: "FFFFFF", align: "center", valign: "middle" });
  text(title, { x: x + 0.92, y: y - 0.02, w: COL - 0.92, h: 0.76, fontSize: 34, bold: true, color: NAVY, valign: "middle" });
}
function subheading(i, t, y) {
  text(t, { x: colX(i), y, w: COL, h: 0.55, fontSize: 26, bold: true, color: INK, valign: "top" });
}
function figure(i, file, y, aspect, alt) {
  const h = COL * aspect;
  s.addImage({ path: img(file), x: colX(i), y, w: COL, h, altText: alt });
  return y + h;
}
function caption(i, t, y, h) {
  text(t, { x: colX(i), y, w: COL, h, fontSize: 18, color: INK2, valign: "top" });
}
const TOP = 4.15;

// 1 · Why
heading(0, 1, "Why", TOP);
text([
  { text: "Maps of the bed beneath the ice need ", options: {} },
  { text: "many equally likely versions", options: { bold: true } },
  { text: " (realizations) to show what radar data can't pin down. GStatSim makes them with sequential Gaussian simulation (SGS): trusted, but minutes per map on a CPU.", options: {} },
], { x: colX(0), y: TOP + 1.0, w: COL, h: 2.6, fontSize: 24, color: INK, valign: "top" });
text([{ text: "Goal: ", options: { bold: true, color: BLUE } },
      { text: "run GStatSim on a GPU without changing a single answer.", options: { color: INK } }],
  { x: colX(0), y: TOP + 3.75, w: COL, h: 1.0, fontSize: 24, valign: "top" });
let y1 = figure(0, "study_area.png", TOP + 5.0, 1.089, "Map of radar bed data around Rubble Glacier");
caption(0, "Study area: Rubble Glacier, George VI Sound, Antarctic Peninsula. Radar bed (Bedmap3 thickness, BedMachine v3 surface) and exposed rock, 500 m cells. Polar stereographic, north up.", y1 + 0.12, 1.3);

// 2 · How
heading(1, 2, "How", TOP);
let y2 = figure(1, "workflow.png", TOP + 1.0, 1.002, "Workflow and CUDA kernel diagram");
subheading(1, "Why keep the exact order?", y2 + 0.35);
y2 = figure(1, "batched.png", y2 + 0.95, 0.649, "Variograms of exact vs whole-grid-at-once SGS");
caption(1, "Simulating the whole grid at once, a common GPU shortcut, makes the bed 39% rougher at 1 km than the data support. Exact SGS does not.", y2 + 0.12, 1.0);

// 3 · Results
heading(2, 3, "Rubble Glacier bed", TOP);
let y3 = figure(2, "realizations.png", TOP + 1.0, 1.080, "Four SGS realizations of the bed");
caption(2, "Four of 20 GPU realizations (1 km grid). Each honors the radar data and its variogram; where they differ, the bed is uncertain.", y3 + 0.1, 0.95);
y3 = figure(2, "uncertainty.png", y3 + 1.15, 0.623, "Ensemble uncertainty and difference from BedMachine");
caption(2, "Left: spread of the 20 realizations, near zero along radar lines. Right: ensemble mean minus BedMachine v3.", y3 + 0.1, 0.95);

// 4 · Is it right? How fast?
heading(3, 4, "Same answers, much faster", TOP);
let y4 = figure(3, "validation.png", TOP + 1.0, 0.631, "GPU vs GStatSim agreement");
caption(3, "Left: 92% of cells match to rounding; 8% are \"tied\" cells (two data points equally far), where GStatSim's pick depends on the CPU. Right: SGS ensembles agree as closely as two GStatSim ensembles do.", y4 + 0.1, 1.3);
y4 = figure(3, "amundsen.png", y4 + 1.15, pngAspect("amundsen.png"),
  "Kriged bed of the Amundsen Sea Embayment, Thwaites and Pine Island glaciers");
y4 = figure(3, "speed.png", y4 + 0.15, pngAspect("speed.png"), "Run times, GStatSim vs GPU");
console.log(`column 4 ends at ${y4.toFixed(2)} (frame units; conclusions band at 21.55)`);

// 5 · Conclusions & next steps (full-width band)
const CY = 21.55, CH = 2.75;
s.addShape(pres.shapes.RECTANGLE, { x: M, y: CY, w: 36 - 2 * M, h: CH, fill: { color: TINT }, line: { color: TINT } });
s.addShape(pres.shapes.OVAL, { x: M + 0.3, y: CY + 0.3, w: 0.72, h: 0.72, fill: { color: NAVY }, line: { color: NAVY } });
text("5", { x: M + 0.3, y: CY + 0.3, w: 0.72, h: 0.72, fontSize: 30, bold: true, color: "FFFFFF", align: "center", valign: "middle" });
text("Conclusions & next steps", { x: M + 1.22, y: CY + 0.28, w: 12, h: 0.76, fontSize: 34, bold: true, color: NAVY, valign: "middle" });
const findings = [
  ["The bigger the map, the bigger the win.",
    "Rubble Glacier kriging is 62× faster; the 1.15-million-cell Amundsen grid is 162×. A map that took GStatSim 4.7 h takes under 2 min."],
  ["Exact SGS is worth keeping.",
    "Simulating the whole grid at once is easier on a GPU but makes the bed 39% rougher than the data support. One GPU block per realization keeps GStatSim's exact algorithm."],
  ["Ties can match exactly too.",
    "With a stable sort in GStatSim's neighbor search, every tested cell and SGS node matches to 1e-14 (was 8% / 20% differing), and GStatSim gives the same answer on every CPU."],
  ["Next: into GStatSim.",
    "gstatsim_gpu.py + GPU tutorials go to GStatSim and the GStatSim Jupyter Book; stable-sort fix proposed upstream; large-region GPU vs CPU tutorial."],
];
const FW = (36 - 2 * M - 0.6 - 3 * 0.5) / 4;
findings.forEach(([head, body], k) => {
  text([
    { text: head, options: { bold: true, color: k < 3 ? INK : BLUE, breakLine: true } },
    { text: body, options: { color: INK2 } },
  ], { x: M + 0.3 + k * (FW + 0.5), y: CY + 1.2, w: FW, h: CH - 1.35, fontSize: 20, valign: "top", paraSpaceAfter: 4 });
});

function pngAspect(name) {
  // read the PNG header for its size
  const b = fs.readFileSync(img(name));
  return b.readUInt32BE(20) / b.readUInt32BE(16);
}

// ------------------------------------------------------------------ footer
const FY = 24.75;
s.addShape(pres.shapes.LINE, { x: M, y: FY - 0.2, w: 36 - 2 * M, h: 0, line: { color: "E1E0D9", width: 1.5 } });
const stats = [
  [`${xKrig}×`, "faster kriging, 1.15 million\ncells (Amundsen, 500 m)"],
  [`${xSgsMany}×`, "faster simulation, same grid\n(per realization, 32 at once)"],
  ["1e-14", "largest difference from GStatSim\nwith a stable sort (proposed fix)"],
  ["7", "GStatSim book chapters\nnow run on the GPU"],
];
stats.forEach(([big, small], k) => {
  const x = M + k * 4.6;
  text(big, { x, y: FY, w: 4.4, h: 1.05, fontSize: 60, bold: true, color: k < 2 ? BLUE : NAVY, valign: "top" });
  text(small, { x, y: FY + 1.08, w: 4.4, h: 0.8, fontSize: 16, color: INK2, valign: "top" });
});
text([
  { text: "References  ", options: { bold: true, color: INK } },
  { text: "MacKie et al. (2023) GStatSim V1.0, Geosci. Model Dev. 16, 3765–3783 · Morlighem (2022) BedMachine Antarctica v3, NSIDC · Pritchard et al. (2025) Bedmap3, Sci. Data 12, 414 · Okuta et al. (2017) CuPy, LearningSys/NeurIPS.", options: { color: INK2, breakLine: true } },
  { text: "Upstream fixes  ", options: { bold: true, color: INK } },
  { text: "GatorGlaciology/GStatSim#30 (SGS on NumPy ≥ 2.4, in 1.2.1) · #31 (rbf_trend on non-square grids) · gstatsimbook#7 (chapters 3–9). Computing: UFIT Research Computing, HiPerGator (NVIDIA L4); development on a laptop RTX 5060.", options: { color: INK2 } },
], { x: 19.4, y: FY, w: 12.6, h: 1.9, fontSize: 14, valign: "top", paraSpaceAfter: 6 });
s.addImage({ path: img("qr_repo.png"), x: 32.95, y: FY - 0.05, w: 1.6, h: 1.6, altText: "QR code linking to the GitHub repository" });
text("Code & notebooks", { x: 32.4, y: FY + 1.58, w: 2.7, h: 0.4, fontSize: 13, color: INK2, align: "center" });

const out = path.join(HERE, "poster.pptx");
pres.writeFile({ fileName: out }).then(() => console.log("wrote", out));
