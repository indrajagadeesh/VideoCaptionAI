from typing import Dict
from ..models.ass_settings import AssSettings

def convert_json_to_ass(data: Dict, ass_settings: AssSettings) -> str:
    """
    Convert JSON (with segments & words) to an ASS karaoke file.
    """
    def format_time(sec: float) -> str:
        # H:MM:SS.cs
        h = int(sec // 3600)
        m = int((sec % 3600) // 60)
        s = sec % 60
        s_int = int(s)
        cs = int(round((s - s_int)*100))  # centiseconds
        return f"{h}:{m:02}:{s_int:02}.{cs:02}"

    lines = []
    # [Script Info]
    lines.append("[Script Info]")
    lines.append("Title: Karaoke Subtitles")
    lines.append("ScriptType: v4.00+")
    lines.append("PlayResX: 1280")
    lines.append("PlayResY: 720")
    lines.append("Timer: 100.0")
    lines.append("")

    # [V4+ Styles]
    lines.append("[V4+ Styles]")
    lines.append("Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour,"
                 " OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut,"
                 " ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow,"
                 " Alignment, MarginL, MarginR, MarginV, Encoding")

    style_str = (
        f"Style: {ass_settings.Name},{ass_settings.Fontname},{ass_settings.Fontsize},"
        f"{ass_settings.PrimaryColour},{ass_settings.SecondaryColour},{ass_settings.OutlineColour},"
        f"{ass_settings.BackColour},{ass_settings.Bold},{ass_settings.Italic},{ass_settings.Underline},"
        f"{ass_settings.StrikeOut},{ass_settings.ScaleX},{ass_settings.ScaleY},{ass_settings.Spacing},"
        f"{ass_settings.Angle},{ass_settings.BorderStyle},{ass_settings.Outline},{ass_settings.Shadow},"
        f"{ass_settings.Alignment},{ass_settings.MarginL},{ass_settings.MarginR},{ass_settings.MarginV},"
        f"{ass_settings.Encoding}"
    )
    lines.append(style_str)
    lines.append("")

    # [Events]
    lines.append("[Events]")
    lines.append("Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text")

    # Build lines
    for seg in data["segments"]:
        start_time = format_time(seg["start"])
        end_time = format_time(seg["end"])
        karaoke_parts = []
        for w in seg["words"]:
            duration_cs = int(round((w["end"] - w["start"])*100))
            karaoke_parts.append(f"{{\\k{duration_cs}}}{w['word']}")
        text_line = " ".join(karaoke_parts)
        line_str = f"Dialogue: 0,{start_time},{end_time},{ass_settings.Name},,0,0,0,,{text_line}"
        lines.append(line_str)

    return "\n".join(lines)
