"""Embed data.json into template.html -> index.html (single static file)."""
import pathlib
d = pathlib.Path("data.json").read_text()
t = pathlib.Path("template.html").read_text()
pathlib.Path("index.html").write_text(t.replace("__DATA__", d))
print("index.html", round(len(t) / 1024 + len(d) / 1024), "KB")
