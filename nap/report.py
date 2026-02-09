"""Report — HTML report generator using a template with placeholder markers."""

import base64
import shutil
from datetime import datetime
from pathlib import Path


TEMPLATE_PATH = Path(__file__).parent / "templates" / "report.html"


class Report:
    """Builds an HTML report by replacing placeholders in a template."""

    def __init__(self, figures_dir: str):
        self._template = TEMPLATE_PATH.read_text()
        self._content = self._template
        self._figures_dir = Path(figures_dir)
        self._figures_dir.mkdir(parents=True, exist_ok=True)

    def set(self, key: str, html: str):
        """Replace {{KEY}} placeholder with html content."""
        self._content = self._content.replace("{{" + key + "}}", html)

    def add_image(self, key: str, image_path: str, title: str = "") -> str:
        """Encode image as base64, set it on the key, and copy to figures dir.

        Returns the base64 img tag for reuse.
        """
        path = Path(image_path)
        data = base64.b64encode(path.read_bytes()).decode()
        img_tag = f'<img src="data:image/png;base64,{data}" alt="{title}">'
        if title:
            img_tag = f"<h3>{title}</h3>\n{img_tag}"

        # Copy figure to subject's figures dir
        dest = self._figures_dir / path.name
        if path != dest:
            shutil.copy2(str(path), str(dest))

        self.set(key, img_tag)
        return img_tag

    def append(self, key: str, html: str):
        """Append html content before the {{KEY}} placeholder (keeps placeholder for more appends)."""
        self._content = self._content.replace(
            "{{" + key + "}}",
            html + "\n{{" + key + "}}",
        )

    def append_image(self, key: str, image_path: str, title: str = "") -> str:
        """Encode image and append it before the {{KEY}} placeholder."""
        path = Path(image_path)
        data = base64.b64encode(path.read_bytes()).decode()
        img_tag = f'<img src="data:image/png;base64,{data}" alt="{title}">'
        if title:
            img_tag = f"<h3>{title}</h3>\n{img_tag}"

        dest = self._figures_dir / path.name
        if path != dest:
            shutil.copy2(str(path), str(dest))

        self.append(key, img_tag)
        return img_tag

    def save(self, subject_dir: str):
        """Write the final report, cleaning up any remaining placeholders."""
        # Set timestamp
        self.set("TIMESTAMP", datetime.now().strftime("%Y-%m-%d %H:%M"))

        # Clean up unused placeholders
        import re
        html = re.sub(r"\{\{[A-Z_]+\}\}", "", self._content)

        out_path = Path(subject_dir) / "report.html"
        out_path.write_text(html)
        return str(out_path)
