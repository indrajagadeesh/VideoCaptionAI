from typing import Dict, List
from ..models.ass_settings import AssSettings

def convert_json_to_ass(data: Dict, ass_settings: AssSettings, subtitle_style: str = "classic") -> str:
    """
    Convert a JSON structure (with segments & words) to an ASS subtitle file
    for one of the following styles:
      - classic
      - karaoke
      - highlight
      - underline
      - word_by_word
    """

    def format_time(sec: float) -> str:
        """Convert floating-point seconds to ASS time format H:MM:SS.cc."""
        h = int(sec // 3600)
        m = int((sec % 3600) // 60)
        s = sec % 60
        s_int = int(s)
        cs = int(round((s - s_int) * 100))  # centiseconds
        return f"{h}:{m:02}:{s_int:02}.{cs:02}"

    # Common header lines
    lines: List[str] = []
    lines.append("[Script Info]")
    lines.append("Title: Subtitles")
    lines.append("ScriptType: v4.00+")
    lines.append(f"PlayResX: {ass_settings.PlayResX}")
    lines.append(f"PlayResY: {ass_settings.PlayResY}")
    lines.append("ScaledBorderAndShadow: yes")
    lines.append("")

    # V4+ Styles header
    lines.append("[V4+ Styles]")
    lines.append("Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour,"
                 " OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut,"
                 " ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow,"
                 " Alignment, MarginL, MarginR, MarginV, Encoding")

    # Define our single style (usually "Default")
    style_line = (
        f"Style: {ass_settings.Name},{ass_settings.Fontname},{ass_settings.Fontsize},"
        f"{ass_settings.PrimaryColour},{ass_settings.SecondaryColour},"
        f"{ass_settings.OutlineColour},{ass_settings.BackColour},"
        f"{ass_settings.Bold},{ass_settings.Italic},{ass_settings.Underline},"
        f"{ass_settings.StrikeOut},{ass_settings.ScaleX},{ass_settings.ScaleY},"
        f"{ass_settings.Spacing},{ass_settings.Angle},{ass_settings.BorderStyle},"
        f"{ass_settings.Outline},{ass_settings.Shadow},{ass_settings.Alignment},"
        f"{ass_settings.MarginL},{ass_settings.MarginR},{ass_settings.MarginV},"
        f"{ass_settings.Encoding}"
    )
    lines.append(style_line)
    lines.append("")

    # Events header
    lines.append("[Events]")
    lines.append("Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text")

    # Helper: Position tag if you want \anN + \pos(x,y)
    # (You can omit \anN if you're already setting "Alignment" in the style.)
    position_tag = ""
    # f"\\an{ass_settings.DefaultAlignment}\\pos({ass_settings.PositionX},{ass_settings.PositionY})"

    # Generate events based on style:
    # style = ass_settings.style.lower().strip()

    if subtitle_style == "classic":
        # 1. CLASSIC
        # One line per segment: entire text is shown from segment start to end.
        # "Hello world" from 0:00 to 0:03, "This is a test" from 0:03 to 0:06, etc.
        for seg in data["segments"]:
            start_time = format_time(seg["start"])
            end_time = format_time(seg["end"])
            text = seg["text"]  # full segment text
            # Example: Dialogue: 0,0:00:00.00,0:00:03.00,Default,,0,0,0,,{\an5\pos(640,360)}Hello world
            line_str = (
                f"Dialogue: 0,{start_time},{end_time},{ass_settings.Name},,0,0,0,,"
                f"{{{position_tag}}}{text}"
            )
            lines.append(line_str)

    elif subtitle_style == "karaoke":
        # 2. KARAOKE
        # Each segment is one line, with embedded \k tags for each word's duration in centiseconds.
        for seg in data["segments"]:
            start_time = format_time(seg["start"])
            end_time = format_time(seg["end"])

            # Build karaoke string with \k tags
            karaoke_parts = []
            for w in seg["words"]:
                # Duration in centiseconds
                duration_cs = int(round((w["end"] - w["start"]) * 100))
                # e.g. "{\k150}Hello"
                karaoke_parts.append(f"{{\\k{duration_cs}}}{w['word']}")

            # Join words with a space
            text_line = " ".join(karaoke_parts)
            # Optionally apply word color: {\c&H00FFFF00&} or from ass_settings.WordColour
            # e.g.: {\c&H00FFFF00&}{\k150}Hello {\k150}world
            text_line = f"{{\\c{ass_settings.WordColour}}}{text_line}"

            line_str = (
                f"Dialogue: 0,{start_time},{end_time},{ass_settings.Name},,0,0,0,,"
                f"{{{position_tag}}}{text_line}"
            )
            lines.append(line_str)

    elif subtitle_style == "highlight":
        # 3. HIGHLIGHT
        # For each segment, produce multiple events, one per word.
        # In each event, the "current" word is in the highlight color, and the others are normal color.
        for seg in data["segments"]:
            words = seg["words"]
            # For each word i in the segment:
            for i, w in enumerate(words):
                start_time = format_time(w["start"])
                end_time = format_time(w["end"])

                # Build a line of text for all words, highlighting only the i-th
                text_parts = []
                for j, w2 in enumerate(words):
                    if j == i:
                        # highlight color
                        text_parts.append(f"{{\\c{ass_settings.WordColour}}}{w2['word']}")
                    else:
                        # normal (primary) color
                        text_parts.append(f"{{\\c{ass_settings.PrimaryColour}}}{w2['word']}")

                # Join with spaces
                text_line = " ".join(text_parts)

                line_str = (
                    f"Dialogue: 0,{start_time},{end_time},{ass_settings.Name},,0,0,0,,"
                    f"{{{position_tag}}}{text_line}"
                )
                lines.append(line_str)

    elif subtitle_style == "underline":
        # 4. UNDERLINE
        # Similar to "highlight", but instead of color, we use {\u1} for the active word.
        for seg in data["segments"]:
            words = seg["words"]
            for i, w in enumerate(words):
                start_time = format_time(w["start"])
                end_time = format_time(w["end"])

                text_parts = []
                for j, w2 in enumerate(words):
                    if j == i:
                        # underline on, then off after the word
                        text_parts.append(f"{{\\u1}}{w2['word']}{{\\u0}}")
                    else:
                        text_parts.append(w2['word'])
                text_line = " ".join(text_parts)

                line_str = (
                    f"Dialogue: 0,{start_time},{end_time},{ass_settings.Name},,0,0,0,,"
                    f"{{{position_tag}}}{text_line}"
                )
                lines.append(line_str)

    elif subtitle_style == "word_by_word":
        # 5. WORD-BY-WORD
        # Each word is a standalone event shown for its own duration,
        # in the highlight color (or "word color").
        for seg in data["segments"]:
            for w in seg["words"]:
                start_time = format_time(w["start"])
                end_time = format_time(w["end"])

                # e.g. "{\c&H00FFFF00&}Hello"
                text_line = f"{{\\c{ass_settings.WordColour}}}{w['word']}"

                line_str = (
                    f"Dialogue: 0,{start_time},{end_time},{ass_settings.Name},,0,0,0,,"
                    f"{{{position_tag}}}{text_line}"
                )
                lines.append(line_str)

    else:
        # Fallback (if style is unknown, do "classic")
        for seg in data["segments"]:
            start_time = format_time(seg["start"])
            end_time = format_time(seg["end"])
            text = seg["text"]
            line_str = (
                f"Dialogue: 0,{start_time},{end_time},{ass_settings.Name},,0,0,0,,"
                f"{{{position_tag}}}{text}"
            )
            lines.append(line_str)

    return "\n".join(lines)
