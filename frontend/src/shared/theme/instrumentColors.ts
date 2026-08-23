import type { Family } from "@/entities/project/types";

export type ThemeName = "dark" | "light";

/** 乐器族配色（计划书 §4.4：10 族分色）——深浅两套，浅色版加深保证白底可读 */
export const FAMILY_COLORS: Record<ThemeName, Record<Family, string>> = {
  dark: {
    guitar: "#FF8A5C",
    bass: "#B388FF",
    keys: "#4FC3F7",
    strings: "#81C784",
    winds: "#4DB6AC",
    brass: "#FFD54F",
    synth: "#7986CB",
    vocal: "#F06292",
    perc: "#B0BEC5",
    fx: "#78909C",
  },
  light: {
    guitar: "#D2551E",
    bass: "#6A3ABF",
    keys: "#0277BD",
    strings: "#2E7D32",
    winds: "#00796B",
    brass: "#A87900",
    synth: "#3949AB",
    vocal: "#C2185B",
    perc: "#546E7A",
    fx: "#455A64",
  },
};

export function familyColor(theme: ThemeName, family: Family): string {
  return FAMILY_COLORS[theme][family];
}

export const FAMILY_NAMES: Record<Family, string> = {
  guitar: "吉他",
  bass: "贝斯",
  keys: "键盘",
  strings: "弦乐",
  winds: "管乐",
  brass: "铜管",
  synth: "合成器",
  vocal: "人声",
  perc: "打击乐",
  fx: "其他",
};

/**
 * ia-amt 36 类 instrument_class → 族（阶段二 notes.json 就位后按类着色用）。
 * 类名表取自 score_extraction/external/ia-amt taxonomy/instrument_classes.py
 */
export const CLASS_FAMILY: Record<string, Family> = {
  acoustic_guitar: "guitar",
  electric_guitar_clean: "guitar",
  electric_guitar_muted: "guitar",
  distorted_guitar: "guitar",
  guitar_harmonics: "guitar",
  acoustic_bass: "bass",
  electric_bass: "bass",
  synth_bass: "bass",
  slap_bass: "bass",
  piano: "keys",
  electric_piano: "keys",
  organ: "keys",
  plucked_keyboard: "keys",
  accordion_family: "keys",
  strings: "strings",
  pizzicato_strings: "strings",
  orchestral_harp: "strings",
  orchestral_woodwind: "winds",
  flute_pipe: "winds",
  harmonica: "winds",
  brass: "brass",
  sax: "brass",
  synth_lead: "synth",
  synth_pad: "synth",
  melody: "vocal",
  vocal_harmony: "vocal",
  choir: "vocal",
  drums: "perc",
  timpani: "perc",
  chromatic_percussion: "perc",
  wind_chimes: "perc",
  percussive_fx: "fx",
  sound_fx: "fx",
  synth_fx: "fx",
  orchestra_hit: "fx",
  ethnic: "fx",
};

/** GM 128 音色名（MVP 从 .mid 解析时用） */
export const GM_NAMES: string[] = [
  "钢琴", "亮音钢琴", "电钢琴", "酒吧钢琴", "电钢琴1", "电钢琴2", "羽管键琴", "克拉维钢琴",
  "钢片琴", "钟琴", "八音盒", "颤音琴", "马林巴", "木琴", "管钟", "扬琴",
  "风琴", "打击风琴", "摇滚风琴", "教堂风琴", "簧风琴", "手风琴", "口风琴", "探戈手风琴",
  "尼龙吉他", "钢弦吉他", "爵士吉他", "清音吉他", "闷音吉他", "过载吉他", "失真吉他", "泛音吉他",
  "原声贝斯", "指弹贝斯", "拨片贝斯", "无品贝斯", "击弦贝斯", "合成贝斯1", "合成贝斯2", "小提琴",
  "中提琴", "大提琴", "低音提琴", "颤弓弦乐", "拨奏弦乐", "竖琴", "定音鼓", "弦乐组",
  "慢弦乐", "合成弦乐", "合唱啊", "人声哦", "童声", "管弦齐奏", "小号", "长号",
  "大号", "弱音小号", "圆号", "铜管组", "合成铜管1", "合成铜管2", "高音萨克斯", "中音萨克斯",
  "次中音萨克斯", "上低音萨克斯", "双簧管", "英国管", "大管", "单簧管", "短笛", "长笛",
  "竖笛", "排箫", "吹瓶", "尺八", "哨笛", "奥卡里那", "合成主音1", "合成主音2",
  "合成主音3", "合成主音4", "合成主音5", "合成主音6", "合成主音7", "合成主音8", "合成铺底1", "合成铺底2",
  "合成铺底3", "合成铺底4", "合成铺底5", "合成铺底6", "合成铺底7", "合成铺底8", "雨声", "音轨",
  "水晶", "大气", "明亮", "诡异", "回声", "科幻", "西塔尔", "班卓",
  "三味线", "筝", "卡林巴", "风笛", "提琴", "唢呐", "叮当铃", "阿哥哥",
  "钢鼓", "木鱼", "太鼓", "旋律鼓", "合成鼓", "镲反转", "吉他 fret 噪声", "呼吸声",
  "海浪", "鸟鸣", "电话铃", "直升机", "掌声", "枪声",
];

/** GM program → 乐器族 */
export function gmFamily(program: number, isDrumChannel = false): Family {
  if (isDrumChannel) return "perc";
  if (program <= 7) return "keys";
  if (program <= 15) return "perc";
  if (program <= 23) return "keys";
  if (program <= 31) return "guitar";
  if (program <= 39) return "bass";
  if (program <= 51) return "strings";
  if (program <= 54) return "vocal";
  if (program === 55) return "fx";
  if (program <= 67) return "brass";
  if (program <= 79) return "winds";
  if (program <= 95) return "synth";
  if (program <= 111) return "fx";
  if (program <= 119) return "perc";
  return "fx";
}

/** 文件名提示（管线产物 guitar.mid/piano.mid 等优先于 GM 音色判断） */
export function filenameHint(filename: string): { family: Family; name: string } | null {
  const n = filename.toLowerCase();
  if (/(guitar|gtr|吉[他它])/.test(n)) return { family: "guitar", name: "吉他" };
  if (/(piano|pf\b|钢琴)/.test(n)) return { family: "keys", name: "钢琴" };
  if (/(bass|贝斯)/.test(n)) return { family: "bass", name: "贝斯" };
  if (/(drum|鼓组)/.test(n)) return { family: "perc", name: "鼓组" };
  if (/(vocal|vox|melody|人声)/.test(n)) return { family: "vocal", name: "人声旋律" };
  return null;
}
