# -*- coding: utf-8 -*-
"""Proje Sunum Şablonu.pptx'i detaylı içerikle doldurur (9 -> 12 slayt)."""
import copy
import re

from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.oxml.ns import qn
from pptx.opc.constants import RELATIONSHIP_TYPE as RT
from pptx.enum.text import PP_ALIGN

SRC = "sunum_calisma.pptx"
OUT = "sunum_dolu.pptx"
NAME = "Mehmet Emre ÜNLÜCAN"
FIG = "outputs"
DARK = RGBColor(0x1A, 0x1A, 0x1A)
NAVY = RGBColor(0x00, 0x23, 0x4A)


def enable_bullet(p):
    """Düz metin kutusunda paragrafa madde işareti (•) ekler."""
    pPr = p._p.get_or_add_pPr()
    for tag in ("a:buNone", "a:buChar", "a:buAutoNum", "a:buFont"):
        for e in pPr.findall(qn(tag)):
            pPr.remove(e)
    pPr.set("marL", "274320")
    pPr.set("indent", "-274320")
    pPr.append(pPr.makeelement(qn("a:buFont"), {"typeface": "Arial"}))
    pPr.append(pPr.makeelement(qn("a:buChar"), {"char": "•"}))


def set_bullets(shape, items, size=18, color=DARK, bullet=False):
    tf = shape.text_frame
    tf.word_wrap = True
    tf.clear()
    for i, (text, lvl) in enumerate(items):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.text = text
        p.level = lvl
        p.space_after = Pt(6)
        if bullet:
            enable_bullet(p)
        for r in p.runs:
            r.font.size = Pt(size)
            r.font.color.rgb = color
    return tf


def find(slide, name=None, text=None):
    for sh in slide.shapes:
        if name and sh.name == name and (text is None or (sh.has_text_frame and text in sh.text_frame.text)):
            return sh
        if text and name is None and sh.has_text_frame and text in sh.text_frame.text:
            return sh
    return None


def remove_decoration(slide):
    """Sag-ust dekoratif cizimi kaldirir (gercek figurlere yer acmak icin)."""
    for sh in list(slide.shapes):
        if (sh.shape_type == 13 and sh.left is not None
                and sh.left > Inches(8) and sh.width > Inches(2) and sh.top < Inches(3)):
            sh._element.getparent().remove(sh._element)


def set_title(slide, text):
    sh = find(slide, name="Unvan 1")
    if not sh:
        return
    p = sh.text_frame.paragraphs[0]
    if p.runs:
        p.runs[0].text = " " + text
        for r in p.runs[1:]:
            r.text = ""
    else:
        p.text = " " + text


def duplicate_slide(prs, index):
    src = prs.slides[index]
    new = prs.slides.add_slide(src.slide_layout)
    for sh in list(new.shapes):
        sh._element.getparent().remove(sh._element)
    # Slayt arka planını (<p:bg>) kopyala — yoksa koyu master arka planına düşer.
    src_cSld = src._element.find(qn("p:cSld"))
    new_cSld = new._element.find(qn("p:cSld"))
    src_bg = src_cSld.find(qn("p:bg"))
    if src_bg is not None and new_cSld.find(qn("p:bg")) is None:
        new_cSld.insert(0, copy.deepcopy(src_bg))
    for sh in src.shapes:
        new.shapes._spTree.append(copy.deepcopy(sh._element))
    for old_rid, rel in src.part.rels.items():
        if rel.reltype == RT.IMAGE and not rel.is_external:
            new_rid = new.part.relate_to(rel.target_part, rel.reltype)
            for blip in new.shapes._spTree.iter(qn("a:blip")):
                if blip.get(qn("r:embed")) == old_rid:
                    blip.set(qn("r:embed"), new_rid)
    return new


def reorder(prs, order):
    lst = prs.slides._sldIdLst
    ids = list(lst)
    for el in ids:
        lst.remove(el)
    for i in order:
        lst.append(ids[i])


def fix_pagenums(prs, total):
    for pos, s in enumerate(prs.slides, 1):
        for sh in s.shapes:
            if sh.has_text_frame and re.fullmatch(r"\s*\d+\s*/\s*\d+\s*", sh.text_frame.text):
                for p in sh.text_frame.paragraphs:
                    if p.runs:
                        p.runs[0].text = f"{pos}/{total}"
                        for r in p.runs[1:]:
                            r.text = ""
                break


def add_table(slide, headers, rows, left, top, width, height, fsize=11):
    cols = len(headers)
    gf = slide.shapes.add_table(len(rows) + 1, cols, Inches(left), Inches(top),
                                Inches(width), Inches(height))
    tbl = gf.table
    for j, h in enumerate(headers):
        c = tbl.cell(0, j)
        c.text = h
        for p in c.text_frame.paragraphs:
            p.alignment = PP_ALIGN.CENTER
            for r in p.runs:
                r.font.size = Pt(fsize); r.font.bold = True; r.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
        c.fill.solid(); c.fill.fore_color.rgb = NAVY
    for i, row in enumerate(rows, 1):
        for j, val in enumerate(row):
            c = tbl.cell(i, j)
            c.text = str(val)
            for p in c.text_frame.paragraphs:
                p.alignment = PP_ALIGN.CENTER if j > 0 else PP_ALIGN.LEFT
                for r in p.runs:
                    r.font.size = Pt(fsize); r.font.color.rgb = DARK
    return gf


def main():
    prs = Presentation(SRC)

    # --- 3 yeni slayt olustur (sona eklenir) ---
    duplicate_slide(prs, 4)   # idx9 = Veri Seti (slayt 5 kopyasi)
    duplicate_slide(prs, 5)   # idx10 = Modalite Karsilastirmasi (slayt 6 kopyasi)
    duplicate_slide(prs, 5)   # idx11 = Grad-CAM (slayt 6 kopyasi)
    # Istenen sira
    reorder(prs, [0, 1, 2, 3, 9, 4, 5, 10, 11, 6, 7, 8])

    sl = list(prs.slides)

    # ===== Slayt 1: Kapak =====
    t = find(sl[0], text="PROJE ADI")
    if t:
        run = t.text_frame.paragraphs[0].runs[0]
        run.text = "UYDU GÖRÜNTÜLERİNDEN DEPREM HASAR TESPİTİ"
        run.font.color.rgb = DARK
    sub = find(sl[0], name="Subtitle 2")
    if sub:
        set_bullets(sub, [
            ("233301041 Mehmet Emre ÜNLÜCAN", 0),
            ("Bilgisayar Mühendisliği Uygulamaları", 0),
            ("Danışman: Prof. Dr. Fatih BAŞÇİFTÇİ", 0),
        ], size=20)

    # ===== Slayt 3: Giriş =====
    set_title(sl[2], "Giriş")
    set_bullets(find(sl[2], name="İçerik Yer Tutucusu 2"), [
        ("6 Şubat 2023 Kahramanmaraş depremleri on binlerce binayı etkilemiş; afet sonrası ilk "
         "saatlerde hasarlı binaların hızla belirlenmesi, arama-kurtarma ve kaynak yönlendirmede "
         "hayati önem taşımaktadır.", 0),
        ("Saha ekipleriyle bina bazlı tespit yavaş ve zahmetlidir; uydu görüntüleri ise geniş bir "
         "alanı çok daha hızlı taramayı mümkün kılar.", 0),
        ("Bu projenin amacı: deprem sonrası uydu görüntülerinden her binayı otomatik olarak "
         "'hasarlı' veya 'sağlam' diye sınıflandıran bir derin öğrenme sistemi geliştirmektir.", 0),
        ("Problem türü ikili sınıflandırmadır: girdi bina görüntüsü, çıktı ise binanın hasar "
         "durumudur.", 0),
        ("Kullanılan QuickQuakeBuildings veri seti 4029 bina içerir; ancak yalnızca 169'u hasarlı, "
         "3860'ı sağlamdır — yani sınıflar arasında yaklaşık 1:23'lük güçlü bir dengesizlik vardır.", 0),
        ("Bu dengesizlikte doğruluk (accuracy) yanıltıcıdır; hasarlı binayı kaçırmamak kritik "
         "olduğundan başarı esas olarak recall ve F1 ile ölçülür.", 0),
    ], size=15)

    # ===== Slayt 4: Kaynak Araştırması =====
    set_title(sl[3], "Kaynak Araştırması")
    set_bullets(find(sl[3], name="İçerik Yer Tutucusu 2"), [
        ("Uzaktan algılamada iki ana görüntü türü vardır: optik (RGB; insan gözüyle yorumlanabilir, "
         "çökme ve enkaz görsel olarak belli olur) ve SAR/radar (bulut ve gece koşullarından "
         "bağımsız, ancak 'speckle' gürültüsü içerir ve yorumlanması zordur).", 0),
        ("Literatürde, deprem öncesi ve sonrası görüntüleri karşılaştıran 'değişim tespiti' "
         "yöntemleri yüksek başarı verir; fakat afet sonrası tek görüntüyle çalışmak pratikte "
         "daha hızlı ve uygulanabilirdir.", 0),
        ("Görüntü sınıflandırmada evrişimli sinir ağları (CNN) standart yaklaşımdır; ResNet "
         "mimarisi, artık (residual) bağlantılar sayesinde derin ağların kararlı eğitilmesini "
         "sağlar.", 0),
        ("Az sayıda etiketli veride, ImageNet'te ön-eğitilmiş ağırlıkların yeni göreve aktarıldığı "
         "transfer öğrenme aşırı öğrenmeyi azaltır ve başarıyı artırır.", 0),
        ("Aşırı sınıf dengesizliği için Focal Loss, kolay örneklerin etkisini azaltıp azınlık "
         "(hasarlı) sınıfa odaklanmayı sağlar.", 0),
        ("Kullanılan veri seti (Sun vd., 2024) ilk yüksek çözünürlüklü SAR-optik kıyaslama setidir; "
         "en iyi sonuç, SAR kolunun SAR-alan ön-eğitimli (SAR-HUB) ağırlıklarla beslendiği "
         "optik-SAR füzyonunda raporlanmıştır.", 0),
    ], size=15)

    # ===== Slayt 5: Materyal ve Yöntem — Veri Seti =====
    set_title(sl[4], "Materyal ve Yöntem: Veri Seti")
    remove_decoration(sl[4])
    ph = find(sl[4], name="İçerik Yer Tutucusu 2")
    ph.width = Inches(7.4)
    set_bullets(ph, [
        ("Veri seti, her biri tek bir binaya ait 4029 örnekten oluşur.", 0),
        ("Her bina için dört dosya bulunur: optik RGB görüntü, optik bina maskesi (footprint), "
         "SAR yoğunluk görüntüsü ve SAR maskesi. Dosyalar MATLAB v7.3 (HDF5) biçimindedir.", 0),
        ("Bina yamalarının boyutları değişkendir (optik ~50–130 piksel, SAR ~135–210 piksel); "
         "bu yüzden tüm görüntüler sabit 224×224 boyutuna ölçeklenir.", 0),
        ("Sınıf dağılımı oldukça dengesizdir: 169 hasarlı (%4,2) ve 3860 sağlam (%95,8) bina.", 0),
        ("Ön işleme: optik görüntü 3 kanallı RGB olarak ImageNet ortalama/standart sapmasıyla; "
         "SAR ise dB değerlerinden [0,1] aralığına normalize edilir.", 0),
    ], size=15)
    add_table(sl[4], ["Sınıf", "Bina Sayısı", "Oran"],
              [["Hasarlı (damaged)", "169", "%4,2"],
               ["Sağlam (intact)", "3860", "%95,8"],
               ["Toplam", "4029", "%100"]],
              left=8.2, top=2.6, width=4.6, height=1.7, fsize=12)

    # ===== Slayt 6: Materyal ve Yöntem — Model ve Eğitim =====
    set_title(sl[5], "Materyal ve Yöntem: Model ve Eğitim")
    remove_decoration(sl[5])
    ph = find(sl[5], name="İçerik Yer Tutucusu 2")
    ph.height = Inches(2.7)
    set_bullets(ph, [
        ("Model: ImageNet ön-eğitimli ResNet-18; son katman iki sınıfa uyarlanır ve tüm ağ "
         "ince-ayarlanır (transfer öğrenme).", 0),
        ("Sınıf dengesizliği üç yolla ele alınır: ağırlıklı örnekleme, Focal Loss ve recall-odaklı "
         "eşik seçimi.", 0),
        ("İki aşamalı ince-ayar: önce sınıflandırıcı başlığı, sonra düşük öğrenme oranıyla tüm ağ "
         "eğitilir (az veride aşırı öğrenmeyi azaltır).", 0),
        ("Eğitim RTX 3060 GPU'da; veri artırma, TTA ve erken durdurma. Değerlendirme 5-katlı "
         "çapraz doğrulama ile yapılır.", 0),
    ], size=14)
    sl[5].shapes.add_picture(f"{FIG}/training_curves.png", Inches(3.55), Inches(4.55), width=Inches(6.2))

    # ===== Slayt 7: Araştırma Sonuçları — Optik Model =====
    set_title(sl[6], "Araştırma Sonuçları: Optik Model")
    remove_decoration(sl[6])
    ph = find(sl[6], name="İçerik Yer Tutucusu 2")
    ph.width = Inches(5.6)
    set_bullets(ph, [
        ("Optik ResNet-18 modeli, 5-katlı çapraz doğrulamada şu sonuçları verdi: recall 0,69 | "
         "precision 0,57 | F1 0,62 | ROC-AUC 0,95 | PR-AUC 0,67.", 0),
        ("PR-AUC değeri 0,67; rastgele tahminin temel çizgisinin (~0,04) yaklaşık 17 katıdır — "
         "model aşırı dengesizliğe rağmen gerçekten ayırt edebiliyor.", 0),
        ("Hata matrisi (yanda): hasarlı binaların çoğu doğru yakalanırken sınırlı sayıda yanlış "
         "alarm oluşur.", 0),
        ("ROC-AUC 0,95, karar eşiğinden bağımsız olarak yüksek ayırt etme gücünü gösterir.", 0),
    ], size=15)
    sl[6].shapes.add_picture(f"{FIG}/cv_confusion_matrix.png", Inches(7.35), Inches(1.5), width=Inches(3.3))
    sl[6].shapes.add_picture(f"{FIG}/cv_pr_roc.png", Inches(6.55), Inches(4.7), width=Inches(5.9))

    # ===== Slayt 8: Araştırma Sonuçları — Modalite Karşılaştırması =====
    set_title(sl[7], "Araştırma Sonuçları: Modalite Karşılaştırması")
    remove_decoration(sl[7])
    ph = find(sl[7], name="İçerik Yer Tutucusu 2")
    ph.height = Inches(2.4)
    set_bullets(ph, [  # noqa
        ("Tüm modaliteler ve mimariler aynı 5-katlı çapraz doğrulama altyapısında karşılaştırıldı.", 0),
        ("SAR tek başına zayıf ve kararsızdır (F1 0,20); ImageNet füzyonu optiği geçemedi — zayıf "
         "SAR özellikleri güçlü optik özelliklerini seyreltti.", 0),
        ("SAR-HUB (SAR-alan ön-eğitimi) füzyonu en yüksek recall (0,73) ve PR-AUC (0,69) değerini "
         "verdi; footprint ve ResNet-34 ise kazanım sağlamadı (veri tavanı: yalnızca 169 hasarlı).", 0),
    ], size=14)
    add_table(sl[7],
              ["Model", "Recall", "Precision", "F1", "ROC-AUC", "PR-AUC"],
              [["Optik (ResNet-18)", "0,69", "0,57", "0,62", "0,95", "0,67"],
               ["SAR (ResNet-18)", "0,32", "0,29", "0,20", "0,80", "0,24"],
               ["Füzyon (ImageNet)", "0,68", "0,55", "0,60", "0,94", "0,67"],
               ["Füzyon (SAR-HUB)", "0,73", "0,50", "0,59", "0,94", "0,69"],
               ["Optik + Footprint", "0,64", "0,57", "0,60", "0,94", "0,69"],
               ["Optik (ResNet-34)", "0,66", "0,57", "0,61", "0,95", "0,68"]],
              left=1.0, top=4.35, width=11.3, height=2.5, fsize=13)

    # ===== Slayt 9: Araştırma Sonuçları — Açıklanabilirlik (Grad-CAM) =====
    set_title(sl[8], "Araştırma Sonuçları: Açıklanabilirlik (Grad-CAM)")
    remove_decoration(sl[8])
    ph = find(sl[8], name="İçerik Yer Tutucusu 2")
    ph.height = Inches(2.3)
    set_bullets(ph, [
        ("Grad-CAM, modelin kararını verirken görüntünün hangi bölgesine 'baktığını' bir ısı "
         "haritasıyla görselleştirir.", 0),
        ("Aşağıda üst sıra girdi optik görüntülerini, alt sıra ise modelin hasarlı sınıfı için "
         "odaklandığı bölgelerin ısı haritalarını gösterir.", 0),
        ("Isı haritaları bina ve enkaz bölgelerine odaklanır; yani model arka plana değil yapısal "
         "hasar işaretlerine bakmaktadır. Bu, sonuçların açıklanabilir ve güvenilir olduğunu "
         "destekler.", 0),
    ], size=15)
    sl[8].shapes.add_picture(f"{FIG}/gradcam.png", Inches(2.65), Inches(4.35), width=Inches(8.0))

    # ===== Slayt 10: Sonuçlar ve Öneriler =====
    set_title(sl[9], "Sonuçlar ve Öneriler")
    box = find(sl[9], name="Metin kutusu 5")
    box.width = Inches(8.8)
    box.height = Inches(4.9)
    set_bullets(box, [
        ("Deprem sonrası optik görüntü, ayırt edici bilginin büyük bölümünü taşımaktadır; transfer "
         "öğrenme sayesinde yalnızca 169 hasarlı örnekle bile makul başarı elde edilmiştir.", 0),
        ("Recall-odaklı strateji (Focal Loss + dengeli örnekleme + eşik seçimi) ile hasarlı "
         "binaların tespiti önceliklendirilmiş; en iyi recall (0,73) SAR-alan ön-eğitimli "
         "(SAR-HUB) füzyonla elde edilmiştir.", 0),
        ("Modalite ablasyonu dürüst bir bulgu sunar: SAR tek başına, footprint ve daha büyük model "
         "(ResNet-34) optik temel modeli geçememiştir — sınır mimari değil, veri miktarıdır.", 0),
        ("Öneriler: SAR-HUB'a uygun dB normalizasyonu, deprem öncesi/sonrası değişim tespiti, daha "
         "fazla hasarlı örnek toplanması, dikkat (attention) tabanlı füzyon ve model topluluğu "
         "(ensemble) ile başarının artırılması.", 0),
    ], size=16, bullet=True)

    # ===== Slayt 11: Kaynaklar (soyada göre alfabetik) =====
    set_title(sl[10], "Kaynaklar")
    box = sl[10].shapes.add_textbox(Inches(0.6), Inches(1.6), Inches(8.7), Inches(5.2))
    set_bullets(box, [
        ("Deng, J., Dong, W., Socher, R. vd. (2009). ImageNet: A Large-Scale Hierarchical Image "
         "Database. IEEE CVPR, 248-255.", 0),
        ("He, K., Zhang, X., Ren, S., Sun, J. (2016). Deep Residual Learning for Image Recognition. "
         "IEEE CVPR, 770-778.", 0),
        ("Lin, T.-Y., Goyal, P., Girshick, R., He, K., Dollar, P. (2017). Focal Loss for Dense "
         "Object Detection. IEEE ICCV, 2980-2988.", 0),
        ("Paszke, A., Gross, S., Massa, F. vd. (2019). PyTorch: An Imperative Style, "
         "High-Performance Deep Learning Library. NeurIPS, 32.", 0),
        ("Sun, Y., Wang, Y., Eineder, M. (2024). QuickQuakeBuildings: Post-earthquake SAR-Optical "
         "Dataset for Quick Damaged-Building Detection. IEEE Geoscience and Remote Sensing "
         "Letters, 21, 1-5.", 0),
        ("Yang, H., Kang, J., Wang, Y. vd. (2023). SAR-HUB: Pre-training, Fine-tuning, and "
         "Explaining. Remote Sensing, 15(23), 5534.", 0),
    ], size=14)

    # --- Sayfa numaralari ve alt bilgi adi ---
    fix_pagenums(prs, len(sl))
    for s in prs.slides:
        for sh in s.shapes:
            if sh.has_text_frame:
                for p in sh.text_frame.paragraphs:
                    for r in p.runs:
                        if "Adınız ve Soyadınız" in r.text:
                            r.text = r.text.replace("Adınız ve Soyadınız", NAME)

    prs.save(OUT)
    print("kaydedildi:", OUT, "| slayt sayısı:", len(list(prs.slides)))


if __name__ == "__main__":
    main()
