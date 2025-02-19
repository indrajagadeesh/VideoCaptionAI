from pydantic import BaseModel, Field

class AssSettings(BaseModel):
    """ASS subtitle settings with default values for common use cases."""
    
    # Script Info
    PlayResX: int = Field(default=1280, description="Video width resolution")
    PlayResY: int = Field(default=720, description="Video height resolution")
    
    # Style properties
    Name: str = Field(default="Default", description="Style name")
    Fontname: str = Field(default="Arial", description="Font family name")
    Fontsize: int = Field(default=48, description="Font size in points")
    
    # Colors (in ASS hex format &HAABBGGRR)
    PrimaryColour: str = Field(default="&H00FFFFFF", description="Main text color (white)")
    SecondaryColour: str = Field(default="&H00FFFF00", description="Secondary/karaoke color (yellow)")
    OutlineColour: str = Field(default="&H00000000", description="Outline color (black)")
    BackColour: str = Field(default="&H80000000", description="Background color (semi-transparent black)")
    WordColour: str = Field(default="&H0000FFFF", description="Highlight color for words (yellow)")
    
    # Style flags (0=false, 1=true)
    Bold: int = Field(default=0, description="Bold text")
    Italic: int = Field(default=0, description="Italic text")
    Underline: int = Field(default=0, description="Underlined text")
    StrikeOut: int = Field(default=0, description="Strikethrough text")
    
    # Scaling and positioning
    ScaleX: int = Field(default=100, description="Horizontal scaling")
    ScaleY: int = Field(default=100, description="Vertical scaling")
    Spacing: int = Field(default=0, description="Letter spacing")
    Angle: int = Field(default=0, description="Rotation angle")
    BorderStyle: int = Field(default=1, description="Border style (1=outline+shadow, 3=opaque box)")
    Outline: int = Field(default=2, description="Outline width")
    Shadow: int = Field(default=2, description="Shadow distance")
    
    # Alignment and margins
    Alignment: int = Field(default=2, description="Text alignment (2=bottom center)")
    MarginL: int = Field(default=10, description="Left margin")
    MarginR: int = Field(default=10, description="Right margin")
    MarginV: int = Field(default=50, description="Vertical margin")
    
    # Encoding
    Encoding: int = Field(default=1, description="Character encoding")
    
    # Position override (optional)
    DefaultAlignment: int = Field(default=5, description="Override alignment ")
    PositionX: int = Field(default=640, description="X position for centered text")
    PositionY: int = Field(default=360, description="Y position for centered text")

    class Config:
        """Pydantic model configuration."""
        validate_assignment = True
