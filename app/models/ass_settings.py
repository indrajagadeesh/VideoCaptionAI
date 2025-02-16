from pydantic import BaseModel

class AssSettings(BaseModel):
    Name: str = "Karaoke"
    Fontname: str = "Arial"
    Fontsize: int = 50
    PrimaryColour: str = "&H00FFFFFF"
    SecondaryColour: str = "&H000000FF"
    OutlineColour: str = "&H00000000"
    BackColour: str = "&H64000000"
    Bold: int = 0
    Italic: int = 0
    Underline: int = 0
    StrikeOut: int = 0
    ScaleX: int = 100
    ScaleY: int = 100
    Spacing: int = 0
    Angle: int = 0
    BorderStyle: int = 1
    Outline: int = 2
    Shadow: int = 0
    Alignment: int = 2
    MarginL: int = 10
    MarginR: int = 10
    MarginV: int = 10
    Encoding: int = 1
