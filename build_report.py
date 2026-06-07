# -*- coding: utf-8 -*-
"""Selçuk Üniversitesi şablonunu (rapor_unpacked) gerçek proje içeriğiyle doldurur."""
import re

DOC = "rapor_unpacked/word/document.xml"
RELS = "rapor_unpacked/word/_rels/document.xml.rels"


def esc(t):
    return t.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def para(text, style="TezMetni15Satr", bold=False, center=False):
    rpr = "<w:rPr><w:b/></w:rPr>" if bold else ""
    jc = '<w:jc w:val="center"/>' if center else ""
    return (f'<w:p><w:pPr><w:pStyle w:val="{style}"/>{jc}</w:pPr>'
            f'<w:r>{rpr}<w:t xml:space="preserve">{esc(text)}</w:t></w:r></w:p>')


def empty(style="TezMetni15Satr"):
    return f'<w:p><w:pPr><w:pStyle w:val="{style}"/></w:pPr></w:p>'


def h1(text):
    return (f'<w:p><w:pPr><w:pStyle w:val="Balk1derece"/><w:pageBreakBefore/></w:pPr>'
            f'<w:r><w:t xml:space="preserve">{esc(text)}</w:t></w:r></w:p>')


def h2(text):
    return (f'<w:p><w:pPr><w:pStyle w:val="Balk2derece"/></w:pPr>'
            f'<w:r><w:t xml:space="preserve">{esc(text)}</w:t></w:r></w:p>')


def h3(text):
    return (f'<w:p><w:pPr><w:pStyle w:val="Balk3derece"/></w:pPr>'
            f'<w:r><w:t xml:space="preserve">{esc(text)}</w:t></w:r></w:p>')


def caption(text):
    return (f'<w:p><w:pPr><w:jc w:val="center"/><w:rPr><w:sz w:val="20"/></w:rPr></w:pPr>'
            f'<w:r><w:rPr><w:sz w:val="20"/></w:rPr>'
            f'<w:t xml:space="preserve">{esc(text)}</w:t></w:r></w:p>')


def figure(rid, cx, cy, did, cap):
    A = "http://schemas.openxmlformats.org/drawingml/2006/main"
    P = "http://schemas.openxmlformats.org/drawingml/2006/picture"
    draw = (f'<w:p><w:pPr><w:jc w:val="center"/></w:pPr><w:r><w:drawing>'
            f'<wp:inline distT="0" distB="0" distL="0" distR="0">'
            f'<wp:extent cx="{cx}" cy="{cy}"/><wp:effectExtent l="0" t="0" r="0" b="0"/>'
            f'<wp:docPr id="{did}" name="Resim {did}"/>'
            f'<wp:cNvGraphicFramePr><a:graphicFrameLocks xmlns:a="{A}" noChangeAspect="1"/></wp:cNvGraphicFramePr>'
            f'<a:graphic xmlns:a="{A}"><a:graphicData uri="{P}">'
            f'<pic:pic xmlns:pic="{P}"><pic:nvPicPr><pic:cNvPr id="{did}" name="Resim {did}"/>'
            f'<pic:cNvPicPr/></pic:nvPicPr>'
            f'<pic:blipFill><a:blip r:embed="{rid}"/><a:stretch><a:fillRect/></a:stretch></pic:blipFill>'
            f'<pic:spPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="{cx}" cy="{cy}"/></a:xfrm>'
            f'<a:prstGeom prst="rect"><a:avLst/></a:prstGeom></pic:spPr></pic:pic>'
            f'</a:graphicData></a:graphic></wp:inline></w:drawing></w:r></w:p>')
    return draw + caption(cap)


def table(headers, rows, widths, cap=None):
    total = sum(widths)

    def cell(text, w, head=False):
        shd = '<w:shd w:val="clear" w:color="auto" w:fill="D9E2F3"/>' if head else ""
        rpr = "<w:rPr><w:b/><w:sz w:val=\"20\"/></w:rPr>" if head else '<w:rPr><w:sz w:val="20"/></w:rPr>'
        return (f'<w:tc><w:tcPr><w:tcW w:w="{w}" w:type="dxa"/>{shd}'
                f'<w:vAlign w:val="center"/></w:tcPr>'
                f'<w:p><w:pPr><w:jc w:val="center"/><w:rPr><w:sz w:val="20"/></w:rPr></w:pPr>'
                f'<w:r>{rpr}<w:t xml:space="preserve">{esc(str(text))}</w:t></w:r></w:p></w:tc>')

    grid = "".join(f'<w:gridCol w:w="{w}"/>' for w in widths)
    body = "<w:tr>" + "".join(cell(h, w, True) for h, w in zip(headers, widths)) + "</w:tr>"
    for r in rows:
        body += "<w:tr>" + "".join(cell(c, w) for c, w in zip(r, widths)) + "</w:tr>"
    b = ('<w:tblBorders>'
         + "".join(f'<w:{e} w:val="single" w:sz="4" w:space="0" w:color="auto"/>'
                   for e in ["top", "left", "bottom", "right", "insideH", "insideV"])
         + '</w:tblBorders>')
    tbl = (f'<w:tbl><w:tblPr><w:tblW w:w="{total}" w:type="dxa"/><w:jc w:val="center"/>'
           f'{b}<w:tblLook w:val="04A0"/></w:tblPr><w:tblGrid>{grid}</w:tblGrid>{body}</w:tbl>')
    out = caption(cap) if cap else ""
    return out + tbl + empty()


def fill_para(s, needle, blocks):
    """needle içeren <w:p>...</w:p>'yi, aynı stilde 'blocks' (metin listesi) ile değiştirir."""
    i = s.find(needle)
    assert i >= 0, f"bulunamadi: {needle[:40]}"
    start = s.rfind("<w:p ", 0, i)
    end = s.find("</w:p>", i) + len("</w:p>")
    style_m = re.search(r'<w:pStyle w:val="([^"]+)"/>', s[start:end])
    style = style_m.group(1) if style_m else "TezMetni15Satr"
    new = "".join(para(b, style=style) for b in blocks)
    return s[:start] + new + s[end:]


# ============================ İÇERİK ============================

OZET = ("Bu çalışmada, 2023 Türkiye-Suriye depremleri sonrası çekilen yüksek çözünürlüklü "
        "uydu görüntülerinden binaların hasar durumunun otomatik olarak sınıflandırılması "
        "amaçlanmıştır. Problem, her bina yamasını hasarlı (damaged) veya sağlam (intact) "
        "olarak etiketleyen bir ikili sınıflandırma görevi olarak ele alınmıştır. Veri seti "
        "olarak, 4029 binadan (169 hasarlı, 3860 sağlam) oluşan ve her bina için optik, SAR "
        "ve bina ayak izi (footprint) görüntülerini içeren açık QuickQuakeBuildings veri seti "
        "kullanılmıştır. Aşırı sınıf dengesizliği (yaklaşık 1:23) nedeniyle tek başına doğruluk "
        "yetersiz kaldığından; precision, recall, F1, ROC-AUC ve PR-AUC metrikleri esas alınmış, "
        "özellikle hasarlı binaların kaçırılmaması için recall önceliklendirilmiştir. Yöntem "
        "olarak ImageNet ön-eğitimli ResNet-18 evrişimli sinir ağı ile transfer öğrenme "
        "uygulanmış; dengesizlik Focal Loss, dengeli örnekleme ve iki aşamalı ince-ayar ile ele "
        "alınmıştır. Modeller, sonuçların güvenilirliği için 5-katlı çapraz doğrulama ile "
        "değerlendirilmiş ve işlem noktası precision-tabanlı bir eşik seçimiyle belirlenmiştir. "
        "Optik görüntüyle elde edilen model F1 0,62; recall 0,69 ve ROC-AUC 0,95 değerlerine "
        "ulaşmıştır. Modalite ablasyonunda SAR tek başına zayıf kalmış; optik-SAR füzyonunda ise "
        "SAR kolunun SAR-alan ön-eğitimli (SAR-HUB) ağırlıklarla beslenmesi recall değerini 0,73 "
        "seviyesine çıkarmıştır. Sonuçlar, deprem sonrası optik görüntünün ayırt edici bilginin "
        "büyük bölümünü taşıdığını ve transfer öğrenmenin sınırlı veriyle bile etkili olduğunu "
        "göstermektedir.")
OZET_KW = ("Anahtar Kelimeler: Bina hasar tespiti, Derin öğrenme, Evrişimli sinir ağı, Sınıf "
           "dengesizliği, Transfer öğrenme, Uzaktan algılama")

ABSTRACT = ("This study addresses the automatic classification of building damage from very "
            "high-resolution post-event satellite imagery acquired after the 2023 Turkiye-Syria "
            "earthquakes. The task is formulated as a binary classification problem labelling each "
            "building patch as damaged or intact. The publicly available QuickQuakeBuildings "
            "dataset is used, comprising 4029 buildings (169 damaged, 3860 intact) with optical, "
            "SAR and footprint patches per building. Because of the severe class imbalance "
            "(about 1:23), accuracy alone is insufficient; precision, recall, F1, ROC-AUC and "
            "PR-AUC are reported, and recall on the damaged class is prioritised so that damaged "
            "buildings are not missed. An ImageNet-pretrained ResNet-18 convolutional neural "
            "network is fine-tuned via transfer learning, and the imbalance is handled with Focal "
            "Loss, balanced sampling and two-stage fine-tuning. Models are evaluated with 5-fold "
            "cross-validation for reliable estimates, and the operating point is set by a "
            "precision-floor threshold. The optical model reaches F1 0.62, recall 0.69 and "
            "ROC-AUC 0.95. In the modality ablation, SAR alone is weak; in optical-SAR fusion, "
            "initialising the SAR branch with SAR-domain pretrained weights (SAR-HUB) raises recall "
            "to 0.73. The results show that the post-event optical patch carries most of the "
            "discriminative information and that transfer learning is effective even with limited "
            "data.")
ABSTRACT_KW = ("Keywords: Building damage assessment, Class imbalance, Convolutional neural "
               "network, Deep learning, Remote sensing, Transfer learning")

ONSOZ = ("Bu çalışma, Selçuk Üniversitesi Teknoloji Fakültesi Bilgisayar Mühendisliği Bölümü "
         "Mühendislik Tasarımı Projesi kapsamında hazırlanmıştır. Çalışmanın amacı, deprem "
         "sonrası uydu görüntülerinden bina bazlı hasar tespitini derin öğrenme yöntemleriyle "
         "gerçekleştiren, çalışan ve raporlanabilir bir sistem geliştirmektir. Proje boyunca "
         "değerli yönlendirmeleri ve desteği için danışman hocam Prof. Dr. Fatih BAŞÇİFTÇİ'ye, "
         "bu süreçte bana destek olan aileme ve arkadaşlarıma teşekkür ederim.")

KISALT = [
    "CNN: Evrişimli Sinir Ağı (Convolutional Neural Network)",
    "CV: Çapraz Doğrulama (Cross-Validation)",
    "dB: Desibel",
    "GPU: Grafik İşlem Birimi (Graphics Processing Unit)",
    "ROC-AUC: ROC Eğrisi Altında Kalan Alan",
    "PR-AUC: Precision-Recall Eğrisi Altında Kalan Alan (Ortalama Kesinlik)",
    "RGB: Kırmızı-Yeşil-Mavi (Red-Green-Blue)",
    "SAR: Sentetik Açıklıklı Radar (Synthetic Aperture Radar)",
    "TTA: Test Zamanı Veri Artırma (Test-Time Augmentation)",
    "VHR: Çok Yüksek Çözünürlük (Very High Resolution)",
]


def build_chapters():
    x = []
    # ---------------- 1. GİRİŞ ----------------
    x.append(h1("1. GİRİŞ"))
    x.append(empty())
    x.append(para("Depremler, kısa sürede geniş bir alanda yaygın yapısal hasara yol açan ve "
        "müdahale süresinin doğrudan insan hayatını etkilediği doğal afetlerdir. 6 Şubat 2023 "
        "tarihinde meydana gelen Kahramanmaraş merkezli depremler, on binlerce binanın hasar "
        "görmesine neden olmuş; afet sonrası ilk saatlerde hangi binaların hasarlı olduğunun "
        "hızla belirlenmesi, arama-kurtarma ve kaynak yönlendirme açısından kritik önem "
        "kazanmıştır. Saha ekipleriyle bina bazlı hasar tespiti zaman alıcı olduğundan, uydu "
        "görüntüleri üzerinden otomatik hasar tespiti, hızlı durumsal farkındalık için önemli "
        "bir alternatif sunmaktadır."))
    x.append(para("Bu proje kapsamında, deprem sonrası yüksek çözünürlüklü uydu görüntülerinden "
        "bina bazlı hasar tespiti, derin öğrenme tabanlı bir ikili sınıflandırma problemi olarak "
        "ele alınmaktadır. Her bina, hasarlı (damaged) veya sağlam (intact) olmak üzere iki "
        "sınıftan birine atanmaktadır."))
    x.append(h2("1.1. Problemin Tanımı ve Önemi"))
    x.append(empty())
    x.append(para("Problem, bir bina yamasına ait uydu görüntüsünün girdi, binanın hasar "
        "durumunun ise çıktı olduğu denetimli bir sınıflandırma görevidir. Veri setinde hasarlı "
        "binalar, sağlam binalara göre çok daha az sayıdadır (169'a karşı 3860). Bu aşırı sınıf "
        "dengesizliği, modelin her şeyi 'sağlam' tahmin ederek yüksek doğruluk elde etmesine yol "
        "açabilir; bu nedenle başarı, yalnızca doğrulukla değil, hasarlı sınıfı için recall ve "
        "F1 gibi metriklerle ölçülmelidir. Afet senaryosunda bir hasarlı binanın gözden "
        "kaçırılması, yanlış alarmdan daha maliyetli olduğu için recall önceliklidir."))
    x.append(h2("1.2. Projenin Amacı ve Kapsamı"))
    x.append(empty())
    x.append(para("Projenin amacı; çalışan, açıklanabilir ve raporlanabilir bir derin öğrenme "
        "modeli geliştirmektir. Bu kapsamda (i) veri yapısı analiz edilmiş, (ii) uygun görüntü "
        "modalitesi seçilmiş, (iii) transfer öğrenme tabanlı bir model eğitilmiş, (iv) sınıf "
        "dengesizliğini ele alan stratejiler uygulanmış ve (v) sonuçlar 5-katlı çapraz doğrulama "
        "ile güvenilir biçimde raporlanmıştır. Ayrıca optik, SAR ve füzyon modaliteleri "
        "karşılaştırmalı olarak değerlendirilmiştir."))
    x.append(h2("1.3. Raporun Organizasyonu"))
    x.append(empty())
    x.append(para("Raporun ikinci bölümünde ilgili literatür ve kullanılan veri seti "
        "özetlenmektedir. Üçüncü bölümde veri seti, ön işleme, model mimarisi ve eğitim yöntemi "
        "anlatılmaktadır. Dördüncü bölümde deneysel sonuçlar ve modalite karşılaştırması "
        "tartışılmakta; beşinci bölümde ise sonuçlar ve öneriler sunulmaktadır."))

    # ---------------- 2. KAYNAK ARAŞTIRMASI ----------------
    x.append(h1("2. KAYNAK ARAŞTIRMASI"))
    x.append(empty())
    x.append(para("Uzaktan algılama görüntüleriyle bina hasar tespiti, son yıllarda derin "
        "öğrenmenin yaygınlaşmasıyla önemli ilerleme kaydeden bir alandır. Bu bölümde, alandaki "
        "temel yaklaşımlar ile bu çalışmada kullanılan veri seti ve yöntemler özetlenmektedir."))
    x.append(h2("2.1. Uydu Görüntüleriyle Hasar Tespiti"))
    x.append(empty())
    x.append(para("Hasar tespitinde optik ve SAR (Sentetik Açıklıklı Radar) olmak üzere iki "
        "temel görüntü modalitesi kullanılır. Optik görüntüler, insan gözüyle yorumlanabilen RGB "
        "bilgisi sunar ve çökme, enkaz gibi hasar belirtilerini görsel olarak taşır. SAR "
        "görüntüleri ise bulut ve gündüz-gece koşullarından bağımsız veri sağlar; ancak 'speckle' "
        "gürültüsü içerir ve yorumlanması daha zordur. Literatürde, deprem öncesi ve sonrası "
        "görüntülerin karşılaştırılmasına (değişim tespiti) dayanan yöntemler başarılı sonuçlar "
        "vermekle birlikte, afet sonrası tek görüntüyle çalışan hızlı tespit senaryoları pratikte "
        "daha kullanışlıdır."))
    x.append(h2("2.2. Derin Öğrenme ve Transfer Öğrenme"))
    x.append(empty())
    x.append(para("Evrişimli sinir ağları (CNN), görüntü sınıflandırmada standart yaklaşım haline "
        "gelmiştir. He ve arkadaşlarının önerdiği ResNet mimarisi, artık (residual) bağlantılar "
        "sayesinde derin ağların kararlı şekilde eğitilebilmesini sağlamıştır. Sınırlı etiketli "
        "veriyle çalışılan durumlarda, ImageNet gibi büyük veri setlerinde ön-eğitilmiş ağların "
        "ağırlıklarının yeni göreve aktarıldığı transfer öğrenme, aşırı öğrenmeyi azaltarak "
        "başarıyı artırır. Aşırı sınıf dengesizliği durumunda ise Lin ve arkadaşlarının önerdiği "
        "Focal Loss, kolay örneklerin katkısını bastırarak azınlık sınıfa odaklanmayı sağlar."))
    x.append(h2("2.3. QuickQuakeBuildings Veri Seti ve Karşılaştırmalı Çalışmalar"))
    x.append(empty())
    x.append(para("Bu çalışmada kullanılan QuickQuakeBuildings veri seti, Sun ve arkadaşları "
        "(2024) tarafından 2023 Türkiye-Suriye depremleri sonrası oluşturulmuş; her bina için "
        "optik ve SAR görüntü yamalarını ve bina ayak izini içeren ilk yüksek çözünürlüklü "
        "SAR-optik kıyaslama veri setidir. İlgili çalışmada, optik görüntülerin SAR'a göre daha "
        "yüksek başarı verdiği, optik-SAR füzyonunun ise en iyi sonucu sağladığı raporlanmıştır. "
        "Füzyonda SAR kolunun, SAR görüntüleriyle ön-eğitilmiş SAR-HUB ağırlıklarıyla "
        "beslenmesinin kritik olduğu belirtilmektedir. Bu rapor, aynı veri seti üzerinde benzer "
        "bir yöntemi yeniden uygulamakta ve modalitelerin katkısını ablasyon yoluyla "
        "incelemektedir."))

    # ---------------- 3. MATERYAL VE YÖNTEM ----------------
    x.append(h1("3. MATERYAL VE YÖNTEM"))
    x.append(empty())
    x.append(para("Bu bölümde kullanılan veri seti, görüntü ön işleme adımları, model mimarisi, "
        "sınıf dengesizliğine yönelik stratejiler, eğitim ayarları ve değerlendirme yöntemi "
        "açıklanmaktadır. Tüm geliştirme Python ve PyTorch ile yapılmış; eğitimler NVIDIA RTX "
        "3060 ekran kartı üzerinde gerçekleştirilmiştir."))
    x.append(h2("3.1. Veri Seti"))
    x.append(empty())
    x.append(para("Veri seti, her biri benzersiz bir binaya ait 4029 örnekten oluşmaktadır. Her "
        "örnek için dört dosya bulunur: optik RGB görüntü (opt), optik bina maskesi (optftp), SAR "
        "yoğunluk görüntüsü (SAR) ve SAR bina maskesi (SARftp). Görüntüler MATLAB v7.3 (HDF5) "
        "formatında saklanmakta ve h5py ile okunmaktadır. Görüntü boyutları binadan binaya "
        "değişmekte (optik yaklaşık 50-130 piksel, SAR yaklaşık 135-210 piksel) olduğundan, "
        "toplu işlem için sabit boyuta (224x224) yeniden ölçeklendirme uygulanmıştır. Sınıfların "
        "dağılımı Çizelge 3.1'de verilmiştir."))
    x.append(table(
        ["Sınıf", "Örnek Sayısı", "Oran"],
        [["Hasarlı (damaged)", "169", "%4,2"],
         ["Sağlam (intact)", "3860", "%95,8"],
         ["Toplam", "4029", "%100"]],
        [3700, 2400, 2400],
        cap="Çizelge 3.1. Veri setindeki sınıf dağılımı (bina sayısı)."))
    x.append(h2("3.2. Ön İşleme ve Veri Artırma"))
    x.append(empty())
    x.append(para("Optik görüntüler 3 kanallı RGB olarak okunmuş, [0,1] aralığına getirilip "
        "ImageNet ortalama ve standart sapmasıyla normalize edilmiştir. SAR görüntüleri "
        "logaritmik (dB) ölçekte olup, her görüntü kendi ortalama ve standart sapmasına göre "
        "normalize edilmiş (z-skor) ve aykırı değerler kırpılmıştır; modele beslenirken üç kanala "
        "çoğaltılmıştır. Eğitim sırasında aşırı öğrenmeyi azaltmak ve özellikle az sayıdaki "
        "hasarlı örneği çoğaltmak amacıyla yatay/dikey çevirme, rastgele döndürme (+/-20 derece) "
        "ve parlaklık/kontrast değişimi gibi veri artırma teknikleri uygulanmıştır."))
    x.append(h2("3.3. Model Mimarisi"))
    x.append(empty())
    x.append(para("Temel model olarak, ImageNet veri setinde ön-eğitilmiş ResNet-18 evrişimli "
        "sinir ağı kullanılmıştır. Ağın son tam-bağlı katmanı, iki sınıflı çıktı verecek şekilde "
        "değiştirilmiş ve tüm ağ yeni göreve göre ince-ayarlanmıştır. Optik-SAR füzyon modelinde "
        "ise iki ayrı ResNet-18 kolu kullanılmış; her kol kendi modalitesini bağımsız işleyerek "
        "512 boyutlu bir öznitelik vektörü üretmiş, bu vektörler birleştirilerek (öznitelik "
        "düzeyinde, geç füzyon) sınıflandırıcıya verilmiştir. SAR kolu için hem ImageNet hem de "
        "SAR-alan ön-eğitimli (SAR-HUB, TerraSAR-X) ağırlıklar denenmiştir."))
    x.append(h2("3.4. Sınıf Dengesizliği ve Kayıp Fonksiyonu"))
    x.append(empty())
    x.append(para("Aşırı sınıf dengesizliği üç ayrı mekanizmayla ele alınmıştır: (i) her "
        "yığının (batch) yaklaşık eşit oranda hasarlı ve sağlam örnek içermesini sağlayan "
        "ağırlıklı rastgele örnekleme, (ii) kolay örneklerin katkısını bastırarak zor ve azınlık "
        "örneğe odaklanan Focal Loss kayıp fonksiyonu ve (iii) karar eşiğinin, recall'i öne çıkaran "
        "bir kriterle ayarlanması. Böylece model, hasarlı binaları kaçırmamaya yönlendirilmiştir."))
    x.append(h2("3.5. Eğitim Ayarları"))
    x.append(empty())
    x.append(para("Modeller, Adam eniyileyici ile 224x224 girdi boyutunda eğitilmiştir. İki "
        "aşamalı ince-ayar uygulanmış; önce yalnızca sınıflandırıcı başlığı, ardından düşük "
        "öğrenme oranıyla tüm ağ eğitilmiştir. Aşırı öğrenmeyi önlemek için doğrulama kümesindeki "
        "F2 skoru izlenerek erken durdurma kullanılmış ve en iyi model kaydedilmiştir. Çıkarım "
        "aşamasında, tahminlerin kararlılığını artırmak amacıyla test zamanı veri artırma (TTA) "
        "uygulanmıştır. Eğitim sırasında doğrulama kaybı ve metriklerinin gelişimi Şekil 3.1'de "
        "örnek olarak gösterilmektedir."))
    x.append(figure("rId18", 4572000, 1662545, 101,
        "Şekil 3.1. Eğitim/doğrulama kayıp eğrileri ve doğrulama metriklerinin epoklara göre değişimi."))
    x.append(h2("3.6. Değerlendirme Yöntemi"))
    x.append(empty())
    x.append(para("Hasarlı sınıfındaki örnek sayısı az olduğundan, tek bir eğitim/test bölmesi "
        "yüksek varyanslı ve güvenilmez sonuçlar üretir. Bu nedenle, modeller katmanlı (stratified) "
        "5-katlı çapraz doğrulama ile değerlendirilmiş ve metrikler ortalama +/- standart sapma "
        "olarak raporlanmıştır. Her katta, eğitim verisinden ayrılan iç-doğrulama kümesi üzerinde "
        "karar eşiği seçilmiş ve sızıntı olmadan kat test kümesine uygulanmıştır. Dengesizlik "
        "nedeniyle accuracy yanında precision, recall, F1, ROC-AUC ve PR-AUC metrikleri esas "
        "alınmıştır."))

    # ---------------- 4. SONUÇLAR VE TARTIŞMA ----------------
    x.append(h1("4. ARAŞTIRMA SONUÇLARI VE TARTIŞMA"))
    x.append(empty())
    x.append(para("Bu bölümde, optik temel modelin sonuçları, modalite ve mimari "
        "karşılaştırması, SAR-HUB ile füzyon, açıklanabilirlik analizi ve genel tartışma "
        "sunulmaktadır."))
    x.append(h2("4.1. Optik Model Sonuçları"))
    x.append(empty())
    x.append(para("ImageNet ön-eğitimli ResNet-18 ile optik görüntüler üzerinde elde edilen "
        "5-katlı çapraz doğrulama sonuçları şunlardır: recall 0,69; precision 0,57; F1 0,62; "
        "ROC-AUC 0,95 ve PR-AUC 0,67. PR-AUC değeri, rastgele tahmin temel çizgisinin (yaklaşık "
        "0,04) yaklaşık 17 katıdır; bu da modelin aşırı dengesizliğe rağmen gerçek ayırt etme "
        "gücüne sahip olduğunu gösterir. Şekil 4.1'de precision-recall ve ROC eğrileri, Şekil "
        "4.2'de ise hata matrisi verilmiştir."))
    x.append(figure("rId19", 4572000, 1787236, 102,
        "Şekil 4.1. Optik model için precision-recall (sol) ve ROC (sağ) eğrileri (OOF tahminleri)."))
    x.append(figure("rId20", 3291840, 2926080, 103,
        "Şekil 4.2. Optik modelin hata matrisi (çapraz doğrulama, birleşik OOF tahminleri)."))
    x.append(h2("4.2. Modalite ve Mimari Karşılaştırması"))
    x.append(empty())
    x.append(para("Optik, SAR ve füzyon modaliteleri ile farklı mimariler aynı 5-katlı çapraz "
        "doğrulama altyapısında karşılaştırılmıştır (Çizelge 4.1). SAR tek başına belirgin şekilde "
        "zayıf ve kararsız kalmıştır (F1 0,20; ROC-AUC 0,80); bunun temel nedeni, veri setinde "
        "yalnızca deprem sonrası tek SAR görüntüsünün bulunması ve değişim bilgisinin olmamasıdır. "
        "Footprint maskesinin ek kanal olarak eklenmesi ve daha büyük bir omurga (ResNet-34) "
        "kullanılması kayda değer bir kazanım sağlamamıştır; bu, başarının model kapasitesiyle "
        "değil, sınırlı hasarlı örnek sayısıyla (veri tavanı) sınırlı olduğunu göstermektedir."))
    x.append(table(
        ["Model", "Recall", "Precision", "F1", "ROC-AUC", "PR-AUC"],
        [["Optik (ResNet-18)", "0,69", "0,57", "0,62", "0,95", "0,67"],
         ["SAR (ResNet-18)", "0,32", "0,29", "0,20", "0,80", "0,24"],
         ["Füzyon (ImageNet)", "0,68", "0,55", "0,60", "0,94", "0,67"],
         ["Füzyon (SAR-HUB)", "0,73", "0,50", "0,59", "0,94", "0,69"],
         ["Optik + Footprint", "0,64", "0,57", "0,60", "0,94", "0,69"],
         ["Optik (ResNet-34)", "0,66", "0,57", "0,61", "0,95", "0,68"]],
        [2853, 1130, 1130, 1130, 1130, 1130],
        cap="Çizelge 4.1. Modalite ve mimari karşılaştırması (5-katlı CV, hasarlı sınıfı)."))
    x.append(h2("4.3. SAR-HUB ile Füzyon"))
    x.append(empty())
    x.append(para("Optik-SAR füzyonunda SAR kolu ImageNet ağırlıklarıyla başlatıldığında, füzyon "
        "optik modeli geçememiştir; çünkü optik-alana ait ImageNet öznitelikleri SAR'ın gürültülü "
        "yapısına uygun değildir. SAR kolu, SAR görüntüleriyle ön-eğitilmiş SAR-HUB (TerraSAR-X) "
        "ağırlıklarıyla başlatıldığında ise recall 0,73 ve PR-AUC 0,69 ile tüm modeller arasında "
        "en yüksek değerlere ulaşmıştır. Bu, recall'in öncelikli olduğu hasar tespiti senaryosu "
        "için en uygun model konumundadır. Çizelge 4.2'de bu çalışmanın sonuçları, veri setini "
        "öneren Sun ve arkadaşları (2024) ile karşılaştırılmaktadır; optik ve SAR sonuçlarımız "
        "literatürle tutarlıdır."))
    x.append(table(
        ["Çalışma", "Optik F1", "SAR F1", "Füzyon F1", "Füzyon ROC-AUC"],
        [["Bu çalışma", "0,62", "0,20", "0,59", "0,94"],
         ["Sun vd. (2024)", "0,605", "0,184", "0,670", "0,962"]],
        [2503, 1500, 1500, 1500, 1500],
        cap="Çizelge 4.2. Bu çalışmanın sonuçlarının literatürle karşılaştırılması."))
    x.append(h2("4.4. Açıklanabilirlik (Grad-CAM)"))
    x.append(empty())
    x.append(para("Modelin kararlarını görsel olarak açıklamak amacıyla Grad-CAM yöntemi "
        "uygulanmıştır. Şekil 4.3'te üst sırada girdi optik görüntüleri, alt sırada ise modelin "
        "hasarlı sınıfı için dikkatini yoğunlaştırdığı bölgeleri gösteren ısı haritaları yer "
        "almaktadır. Isı haritalarının bina ve enkaz bölgelerine odaklanması, modelin arka plan "
        "yerine yapısal hasar işaretlerine baktığını göstermektedir; bu da sonuçların "
        "güvenilirliğini desteklemektedir."))
    x.append(figure("rId21", 4572000, 1524000, 104,
        "Şekil 4.3. Grad-CAM görselleştirmesi: üst sıra girdi görüntüleri, alt sıra hasarlı sınıfı ısı haritaları."))
    x.append(h2("4.5. Tartışma"))
    x.append(empty())
    x.append(para("Elde edilen sonuçlar üç ana çıkarım sunar. Birincisi, deprem sonrası optik "
        "görüntü, ayırt edici bilginin büyük bölümünü taşımaktadır; SAR tek başına veya naif "
        "füzyonla optik modeli geçememektedir. İkincisi, SAR'ın katkı sağlaması için SAR-alan "
        "ön-eğitiminin (SAR-HUB) gerekli olduğu, literatürdeki bulguyla uyumlu şekilde "
        "doğrulanmıştır. Üçüncüsü, başarı esas olarak yalnızca 169 hasarlı örneğin getirdiği veri "
        "tavanıyla sınırlıdır; daha büyük model veya ek maske bilgisi bu tavanı belirgin şekilde "
        "yükseltmemektedir. Metriklerdeki standart sapmalar, test kümesindeki az sayıdaki hasarlı "
        "örnekten kaynaklanan doğal varyansı yansıtmaktadır."))

    # ---------------- 5. SONUÇLAR VE ÖNERİLER ----------------
    x.append(h1("5. SONUÇLAR VE ÖNERİLER"))
    x.append(empty())
    x.append(h2("5.1. Sonuçlar"))
    x.append(empty())
    x.append(para("Bu çalışmada, deprem sonrası uydu görüntülerinden bina bazlı hasar tespiti "
        "için transfer öğrenme tabanlı, çalışan ve açıklanabilir bir derin öğrenme sistemi "
        "geliştirilmiştir. ImageNet ön-eğitimli ResNet-18 ile optik görüntüler üzerinde, 5-katlı "
        "çapraz doğrulamada F1 0,62; recall 0,69 ve ROC-AUC 0,95 elde edilmiştir. Aşırı sınıf "
        "dengesizliği; Focal Loss, dengeli örnekleme ve recall-odaklı eşik seçimiyle başarıyla "
        "ele alınmıştır. Modaliteler karşılaştırıldığında optik görüntünün en bilgilendirici "
        "modalite olduğu, SAR'ın ancak SAR-alan ön-eğitimi (SAR-HUB) ile füzyona katkı sağladığı "
        "ve bu durumda en yüksek recall (0,73) değerine ulaşıldığı gösterilmiştir. Sonuçlar, "
        "sınırlı etiketli veriyle bile transfer öğrenmenin etkili olduğunu ve dengesizlik altında "
        "doğru metrik seçiminin kritik olduğunu ortaya koymaktadır."))
    x.append(h2("5.2. Öneriler"))
    x.append(empty())
    x.append(para("Gelecek çalışmalar için şu yönler önerilebilir: (i) SAR-HUB'ın beklediği dB "
        "normalizasyonunun birebir uygulanmasıyla füzyon başarısının daha da artırılması; (ii) "
        "deprem öncesi ve sonrası görüntülerin birlikte kullanıldığı değişim tespiti yaklaşımları; "
        "(iii) hasarlı örnek sayısının artırılması veya sentetik veri üretimiyle veri tavanının "
        "yükseltilmesi; (iv) dikkat (attention) tabanlı daha gelişmiş füzyon mimarileri ve (v) "
        "model toplulukları (ensemble) ile kararlılığın artırılması. Bu iyileştirmeler, özellikle "
        "hasarlı binaların tespit oranını (recall) yükseltmeye odaklanmalıdır."))
    return "".join(x)


def main():
    s = open(DOC, encoding="utf-8").read()

    s = fill_para(s, "Özet metnini yazmaya buradan başlayınız", [OZET])
    s = fill_para(s, "Anahtar Kelimeler:", [OZET_KW])
    s = fill_para(s, "Türkçe özet metninin İngilizce", [ABSTRACT])
    s = fill_para(s, ">Keywords:", [ABSTRACT_KW])
    s = fill_para(s, "Önsöz metnini yazım kılavuzuna", [ONSOZ])
    s = fill_para(s, "Simgeleri yazmaya buradan başlayınız",
                  ["Bu çalışmada özel bir simge kullanılmamıştır."])
    s = fill_para(s, "Kısaltmaları yazmaya buradan başlayınız", KISALT)

    # Bölüm 1-5: rfind ile GÖVDE başlığını hedefle (TOC girişini değil).
    i_g = s.rfind("<w:t>1. GİRİŞ</w:t>")
    p_g = s.rfind("<w:p ", 0, i_g)
    i_k = s.rfind("<w:t>KAYNAKLAR</w:t>")
    p_k = s.rfind("<w:p ", 0, i_k)
    s = s[:p_g] + build_chapters() + s[p_k:]

    refs = [
        "Sun, Y., Wang, Y., Eineder, M., 2024, QuickQuakeBuildings: Post-earthquake SAR-Optical "
        "Dataset for Quick Damaged-Building Detection, IEEE Geoscience and Remote Sensing Letters, 21, 1-5.",
        "He, K., Zhang, X., Ren, S., Sun, J., 2016, Deep Residual Learning for Image Recognition, "
        "IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 770-778.",
        "Lin, T.-Y., Goyal, P., Girshick, R., He, K., Dollar, P., 2017, Focal Loss for Dense Object "
        "Detection, IEEE International Conference on Computer Vision (ICCV), 2980-2988.",
        "Deng, J., Dong, W., Socher, R., Li, L.-J., Li, K., Fei-Fei, L., 2009, ImageNet: A "
        "Large-Scale Hierarchical Image Database, IEEE Conference on Computer Vision and Pattern "
        "Recognition (CVPR), 248-255.",
        "Yang, H., Kang, J., Wang, Y. ve ark., 2023, SAR-HUB: Pre-training, Fine-tuning, and "
        "Explaining, Remote Sensing, 15(23), 5534.",
        "Gupta, R., Goodman, B., Patel, N. ve ark., 2019, Creating xBD: A Dataset for Assessing "
        "Building Damage from Satellite Imagery, CVPR Workshops, 10-17.",
        "Paszke, A., Gross, S., Massa, F. ve ark., 2019, PyTorch: An Imperative Style, "
        "High-Performance Deep Learning Library, Advances in Neural Information Processing Systems "
        "(NeurIPS), 32, 8024-8035.",
    ]
    s = fill_para(s, "Anonim, 2006, Tarım istatistikleri", refs)
    s = fill_para(s, "Anonymous, 1989, Farm accountancy", [""])

    open(DOC, "w", encoding="utf-8").write(s)

    r = open(RELS, encoding="utf-8").read()
    add = ""
    for rid, img in [("rId18", "image4.png"), ("rId19", "image5.png"),
                     ("rId20", "image6.png"), ("rId21", "image7.png")]:
        if rid not in r:
            add += (f'<Relationship Id="{rid}" Type="http://schemas.openxmlformats.org/'
                    f'officeDocument/2006/relationships/image" Target="media/{img}"/>')
    r = r.replace("</Relationships>", add + "</Relationships>")
    open(RELS, "w", encoding="utf-8").write(r)
    print("document.xml ve rels guncellendi.")


if __name__ == "__main__":
    main()
