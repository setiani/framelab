// docs/recording-guide.md, kept in the bundle (docs/ is not shipped with the app).
import type { Lang } from "../types";

interface Section {
  title: string;
  items: string[];
}

export const RECORDING_GUIDE: Record<Lang, { title: string; sections: Section[] }> = {
  en: {
    title: "How to record an experiment",
    sections: [
      { title: "Every experiment", items: [
        "Keep the phone completely still: use a tripod or lean it on a stack of books.",
        "Point the camera straight at the plane of motion, about 1 to 1.5 m away.",
        "Put a ruler or tape measure in the same plane as the motion, for example taped to the wall directly behind the ball.",
        "Use a plain background, a brightly coloured object and bright light from the front.",
        "Turn off video stabilisation if your phone allows it.",
        "Record 10 to 20 seconds. At the start, say the values out loud: “Tinggi jatuh satu meter, diameter bola enam koma tujuh senti.” You can also say a hypothesis.",
      ] },
      { title: "Free fall", items: [
        "Use slow motion (120 or 240 fps) if your phone has it.",
        "Use a solid, slightly heavy ball, not a ping-pong ball. Drop it from 0.8 to 1.5 m.",
        "Some phones save slow motion at normal playback speed. Then choose the real capture rate in the upload form.",
      ] },
      { title: "Pendulum", items: [
        "30 or 60 fps is enough. Use a small, heavy bob on a thin string.",
        "Keep the swing small (about 15° or less) and record at least 10 swings.",
        "Measure the length from the pivot to the centre of the bob, and say it out loud.",
      ] },
      { title: "Safety", items: [
        "Heights up to 2 m only. Household objects only: no glass, fire or mains electricity.",
        "Children experiment with an adult. When filming children, show hands and apparatus only.",
      ] },
    ],
  },
  id: {
    title: "Cara merekam eksperimen",
    sections: [
      { title: "Semua eksperimen", items: [
        "Pastikan ponsel benar-benar diam: gunakan tripod atau sandarkan pada tumpukan buku.",
        "Arahkan kamera tegak lurus ke bidang gerak, sekitar 1 sampai 1,5 m.",
        "Letakkan penggaris atau meteran pada bidang yang sama dengan gerakan, misalnya ditempel di dinding tepat di belakang bola.",
        "Gunakan latar polos, objek berwarna mencolok, dan cahaya terang dari depan.",
        "Matikan stabilisasi video jika ponsel mengizinkan.",
        "Rekam 10 sampai 20 detik. Di awal, ucapkan nilainya: “Tinggi jatuh satu meter, diameter bola enam koma tujuh senti.” Anda juga bisa mengucapkan hipotesis.",
      ] },
      { title: "Jatuh bebas", items: [
        "Gunakan slow motion (120 atau 240 fps) jika ponsel mendukung.",
        "Gunakan bola padat yang agak berat, bukan bola pingpong. Jatuhkan dari 0,8 sampai 1,5 m.",
        "Sebagian ponsel menyimpan slow motion dengan kecepatan putar normal. Pilih laju rekam aslinya di formulir unggah.",
      ] },
      { title: "Bandul", items: [
        "30 atau 60 fps sudah cukup. Gunakan beban kecil yang berat pada tali tipis.",
        "Jaga ayunan tetap kecil (sekitar 15° atau kurang) dan rekam minimal 10 ayunan.",
        "Ukur panjang dari titik gantung sampai tengah beban, lalu ucapkan.",
      ] },
      { title: "Keselamatan", items: [
        "Ketinggian maksimal 2 m. Hanya benda rumah tangga: tanpa kaca, api, atau listrik PLN.",
        "Anak-anak bereksperimen bersama orang dewasa. Saat merekam anak, tampilkan tangan dan alat saja.",
      ] },
    ],
  },
};
