// Interface language (ID/EN). The report and claim language follow the same choice.
import type { Lang } from "./types";

const KEY = "framelab.lang";

const STRINGS = {
  en: {
    nav_new: "New measurement", nav_series: "Series", backend_ok: "Backend ready",
    backend_down: "Backend not reachable", demo: "Demo mode",
    hero_title: "Film an experiment. Get a measurement you can check.",
    hero_body: "SAM 3.1 tracks the object in every frame, code fits the physics with uncertainty, "
      + "and an independent auditor checks every claim against its evidence.",
    hero_caption: "Synthetic test clip: the tracked ball, its trail and the measured g.",
    upload_title: "Measure a video", drop_here: "Drop a video here or choose a file",
    drop_hint: "MP4 or MOV, 1 to 120 seconds", choose: "Choose file", lang_label: "Report language",
    fps_label: "Slow-motion capture rate", fps_auto: "Auto (from the file)",
    note_label: "Note for the planner (optional)",
    note_placeholder: "For example: the string is 50 cm, measured to the centre of the ball",
    start: "Start measurement", uploading: "Uploading", how_to_record: "How to record",
    recent: "Recent runs", no_runs: "No runs yet. Your measurements will appear here.",
    samples: "Try a recorded sample", sample_note: "Replay of a recorded run (no API calls)",
    series_hint: "Measuring the same pendulum at several lengths?",
    series_open: "Open the series view",
    stage_ingesting: "Ingest", stage_transcribing: "Listen", stage_planning: "Plan",
    stage_segmenting: "Track", stage_measuring: "Measure", stage_analyzing: "Analyse",
    stage_auditing: "Audit", stage_designing: "Design", stage_rendering: "Report",
    stage_done: "Done", stage_failed: "Failed",
    exp_free_fall: "Free fall", exp_pendulum: "Pendulum", exp_unknown: "Unknown experiment",
    exp_none: "Experiment not identified yet",
    v_PASS: "Pass", v_PASS_WITH_WARNINGS: "Pass with warnings",
    v_NEEDS_RERECORD: "Record again", v_REJECT: "Rejected", running: "Running",
    failed: "Failed", cost: "Model cost",
    result: "Result", reference: "Reference", difference: "Difference",
    interval: "95% confidence interval", inside: "The reference lies inside the interval.",
    outside: "The reference lies outside the interval.", hypothesis: "Hypothesis",
    waiting_result: "The result appears when the analysis is done.",
    period_only: "No string length was given, so the period is the result.",
    setup: "Setup", target: "Tracked object", ruler: "Scale reference", spoken: "Spoken values",
    turn: "turn", transcript: "Transcript", no_transcript: "No speech was transcribed.",
    plan_pending: "The planner has not finished yet.",
    video: "Video", overlay_hint: "Annotated video with masks, trail and result",
    original_hint: "Processed input video (the annotated video follows)", frame: "Frame",
    charts: "Trajectory and fit", charts_pending: "The chart appears once the fit is done.",
    trace: "Agent trace", follow: "Follow live", paused: "Paused", events: "events",
    ledger: "Claim ledger", ledger_hint: "Select a claim to see the evidence it rests on.",
    no_claims: "No claims yet.", chain: "Evidence chain", evidence: "Evidence",
    gates: "Quality gates", gate: "Gate", status: "Status", message: "Message",
    next: "Next experiment", changed: "Change", controlled: "Keep the same",
    predictions: "Predictions", protocol: "Steps", safety: "Safety",
    downloads: "Downloads", report_id: "Report (Bahasa Indonesia)", report_en: "Report (English)",
    feedback_pdf: "Recording feedback (PDF)", data_csv: "Data (CSV)", overlay_mp4: "Annotated video",
    feedback: "Record this experiment again", what_failed: "What went wrong", tips: "How to fix it",
    rerun_fps: "Run again with capture rate", rerun_started: "New run started",
    error_title: "This run stopped with an error", not_found: "This run does not exist.",
    back: "Back to measurements",
    series_title: "Pendulum series", series_intro: "Several runs of one pendulum at different "
      + "string lengths. The slope of T₀² against L gives g; the exponent tests T ∝ √L.",
    new_series: "Create series", members: "Runs in this series", length: "Length",
    period0: "Period T₀", verdict: "Verdict", add_run: "Add run", analyze: "Analyse series",
    analyzing: "Analysing", series_pick: "Choose a finished pendulum run",
    series_empty: "Add at least three pendulum runs with different lengths.",
    slope_g: "g from the slope", exponent: "Exponent n", sqrt_law: "T ∝ √L",
    supported: "supported", not_supported: "not supported", series_list: "Your series",
    no_series: "No series yet.", select_runs: "Select pendulum runs for the new series",
    no_pendulum_runs: "No finished pendulum runs with a known length yet.",
    close: "Close", open_run: "Open",
  },
  id: {
    nav_new: "Pengukuran baru", nav_series: "Seri", backend_ok: "Backend siap",
    backend_down: "Backend tidak terjangkau", demo: "Mode demo",
    hero_title: "Rekam eksperimen. Dapatkan hasil ukur yang bisa diperiksa.",
    hero_body: "SAM 3.1 melacak objek di setiap frame, kode mencocokkan model fisika beserta "
      + "ketidakpastiannya, dan auditor independen memeriksa setiap klaim terhadap buktinya.",
    hero_caption: "Klip uji sintetis: bola yang dilacak, jejaknya, dan g terukur.",
    upload_title: "Ukur sebuah video", drop_here: "Letakkan video di sini atau pilih berkas",
    drop_hint: "MP4 atau MOV, 1 sampai 120 detik", choose: "Pilih berkas",
    lang_label: "Bahasa laporan", fps_label: "Laju rekam slow motion",
    fps_auto: "Otomatis (dari berkas)", note_label: "Catatan untuk planner (opsional)",
    note_placeholder: "Misalnya: panjang tali 50 cm, diukur sampai tengah bola",
    start: "Mulai pengukuran", uploading: "Mengunggah", how_to_record: "Cara merekam",
    recent: "Run terbaru", no_runs: "Belum ada run. Hasil pengukuran Anda akan muncul di sini.",
    samples: "Coba contoh rekaman", sample_note: "Pemutaran ulang run terekam (tanpa panggilan API)",
    series_hint: "Mengukur bandul yang sama dengan beberapa panjang tali?",
    series_open: "Buka tampilan seri",
    stage_ingesting: "Impor", stage_transcribing: "Dengar", stage_planning: "Rencana",
    stage_segmenting: "Lacak", stage_measuring: "Ukur", stage_analyzing: "Analisis",
    stage_auditing: "Audit", stage_designing: "Rancang", stage_rendering: "Laporan",
    stage_done: "Selesai", stage_failed: "Gagal",
    exp_free_fall: "Jatuh bebas", exp_pendulum: "Bandul", exp_unknown: "Eksperimen tak dikenal",
    exp_none: "Eksperimen belum dikenali",
    v_PASS: "Lulus", v_PASS_WITH_WARNINGS: "Lulus dengan catatan",
    v_NEEDS_RERECORD: "Rekam ulang", v_REJECT: "Ditolak", running: "Berjalan",
    failed: "Gagal", cost: "Biaya model",
    result: "Hasil", reference: "Referensi", difference: "Selisih",
    interval: "Selang kepercayaan 95%", inside: "Nilai referensi berada di dalam selang.",
    outside: "Nilai referensi berada di luar selang.", hypothesis: "Hipotesis",
    waiting_result: "Hasil muncul setelah analisis selesai.",
    period_only: "Panjang tali tidak diberikan, jadi hasilnya berupa periode.",
    setup: "Persiapan", target: "Objek yang dilacak", ruler: "Referensi skala",
    spoken: "Nilai yang diucapkan", turn: "giliran", transcript: "Transkrip",
    no_transcript: "Tidak ada ucapan yang ditranskripsi.",
    plan_pending: "Planner belum selesai.",
    video: "Video", overlay_hint: "Video beranotasi dengan mask, jejak, dan hasil",
    original_hint: "Video masukan yang sudah diproses (video beranotasi menyusul)",
    frame: "Frame", charts: "Lintasan dan fit",
    charts_pending: "Grafik muncul setelah fit selesai.", trace: "Jejak agen",
    follow: "Ikuti langsung", paused: "Dijeda", events: "event",
    ledger: "Claim ledger", ledger_hint: "Pilih sebuah klaim untuk melihat bukti yang mendasarinya.",
    no_claims: "Belum ada klaim.", chain: "Rantai bukti", evidence: "Bukti",
    gates: "Gerbang kualitas", gate: "Gerbang", status: "Status", message: "Pesan",
    next: "Eksperimen berikutnya", changed: "Ubah", controlled: "Tetap sama",
    predictions: "Prediksi", protocol: "Langkah", safety: "Keselamatan",
    downloads: "Unduhan", report_id: "Laporan (Bahasa Indonesia)", report_en: "Laporan (English)",
    feedback_pdf: "Umpan balik rekaman (PDF)", data_csv: "Data (CSV)",
    overlay_mp4: "Video beranotasi",
    feedback: "Rekam ulang eksperimen ini", what_failed: "Apa yang salah",
    tips: "Cara memperbaikinya", rerun_fps: "Jalankan ulang dengan laju rekam",
    rerun_started: "Run baru dimulai",
    error_title: "Run ini berhenti karena galat", not_found: "Run ini tidak ada.",
    back: "Kembali ke pengukuran",
    series_title: "Seri bandul", series_intro: "Beberapa run dari satu bandul dengan panjang tali "
      + "berbeda. Kemiringan T₀² terhadap L memberi g; eksponennya menguji T ∝ √L.",
    new_series: "Buat seri", members: "Run dalam seri ini", length: "Panjang",
    period0: "Periode T₀", verdict: "Verdict", add_run: "Tambah run", analyze: "Analisis seri",
    analyzing: "Menganalisis", series_pick: "Pilih run bandul yang sudah selesai",
    series_empty: "Tambahkan minimal tiga run bandul dengan panjang berbeda.",
    slope_g: "g dari kemiringan", exponent: "Eksponen n", sqrt_law: "T ∝ √L",
    supported: "didukung", not_supported: "tidak didukung", series_list: "Seri Anda",
    no_series: "Belum ada seri.", select_runs: "Pilih run bandul untuk seri baru",
    no_pendulum_runs: "Belum ada run bandul selesai dengan panjang tali yang diketahui.",
    close: "Tutup", open_run: "Buka",
  },
} as const;

export type Key = keyof (typeof STRINGS)["en"];

export function getLang(): Lang {
  try {
    const v = localStorage.getItem(KEY);
    if (v === "en" || v === "id") return v;
  } catch {
    /* storage unavailable: fall back to the browser language */
  }
  return navigator.language.toLowerCase().startsWith("id") ? "id" : "en";
}

export function setLang(lang: Lang): void {
  try {
    localStorage.setItem(KEY, lang);
  } catch {
    /* ignore */
  }
  document.documentElement.lang = lang;
}

export function t(key: Key, lang: Lang = getLang()): string {
  return STRINGS[lang][key] ?? STRINGS.en[key];
}
