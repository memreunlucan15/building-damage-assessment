# -*- coding: utf-8 -*-
"""PowerPoint COM ile bir .pptx'in her slaydını PNG'ye render eder (görsel kontrol için)."""
import sys
import os
import glob
import win32com.client as win32


def render(pptx_path, out_dir, width=1280):
    os.makedirs(out_dir, exist_ok=True)
    for f in glob.glob(os.path.join(out_dir, "slide-*.png")):
        os.remove(f)
    app = win32.Dispatch("PowerPoint.Application")
    app.Visible = 1
    pres = app.Presentations.Open(os.path.abspath(pptx_path), ReadOnly=1, WithWindow=False)
    h = int(width * pres.PageSetup.SlideHeight / pres.PageSetup.SlideWidth)
    n = pres.Slides.Count
    for i in range(1, n + 1):
        out = os.path.join(out_dir, f"slide-{i:02d}.png")
        pres.Slides(i).Export(os.path.abspath(out), "PNG", width, h)
    pres.Close()
    app.Quit()
    print(f"{n} slayt render edildi -> {out_dir} ({width}x{h})")


if __name__ == "__main__":
    pptx = sys.argv[1] if len(sys.argv) > 1 else r"C:\Users\mehme\Desktop\Proje Sunum Şablonu - Dolu.pptx"
    out = sys.argv[2] if len(sys.argv) > 2 else "render"
    render(pptx, out)
