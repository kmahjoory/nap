"""LLM backend — handles API calls to Claude with text and vision."""

import base64
import time


def _encode_image(image_path: str) -> str:
    """Read an image file and return its base64 encoding."""
    with open(image_path, "rb") as f:
        return base64.standard_b64encode(f.read()).decode("utf-8")


def _call_api(messages: list, max_tokens: int = 2000, retries: int = 5) -> str:
    """Call Claude API with retry on overload/rate limit errors."""
    import anthropic
    client = anthropic.Anthropic()

    for attempt in range(retries):
        try:
            response = client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=max_tokens,
                messages=messages,
            )
            return response.content[0].text
        except (anthropic.APIStatusError, anthropic.APIConnectionError):
            if attempt < retries - 1:
                wait = 3 * (attempt + 1)
                print(f"  API error, retrying in {wait}s...")
                time.sleep(wait)
            else:
                raise


def call_with_images(prompt: str, image_paths: list[str], mock: bool = False) -> str:
    """Send a text prompt with images to Claude and return the response."""
    if mock:
        return _mock_response(image_paths)

    content = []
    for path in image_paths:
        content.append({
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/png",
                "data": _encode_image(path),
            },
        })
    content.append({"type": "text", "text": prompt})

    return _call_api([{"role": "user", "content": content}])


def call_text(prompt: str, mock: bool = False) -> str:
    """Send a text-only prompt to Claude. Returns the response."""
    if mock:
        return "Acknowledged."
    return _call_api([{"role": "user", "content": prompt}], max_tokens=500)


def _mock_response(image_paths: list[str]) -> str:
    """Return a simulated LLM inspection response for development."""
    return (
        "## Visual Inspection Findings (mock mode)\n\n"
        "**Line noise:** Likely 50 Hz line noise visible in raw PSD. "
        "The filtered PSD shows this is removed after bandpass filtering.\n\n"
        "**Bad channels:** Channel Fp2 appears to have elevated broadband noise "
        "compared to neighboring channels, persisting after filtering. "
        "Channel T8 shows intermittent flat segments.\n\n"
        "**Bad segments:** A high-amplitude artifact is visible around 120-125 s "
        "in the raw traces. After filtering, the artifact amplitude is reduced "
        "but still present, suggesting a movement artifact rather than "
        "low-frequency drift.\n\n"
        "**General observations:** The overall data quality appears reasonable. "
        "Alpha activity (~10 Hz) is visible in posterior channels in the PSD. "
        "The filtered PSD shows a clean spectral profile within the passband.\n\n"
        f"*Based on {len(image_paths)} diagnostic plots.*"
    )
