from PIL import Image, ImageDraw
import pdfplumber

from segmentor.document.segment import segment_pdf_page, segment_pdf_image


page_idx = 0
with pdfplumber.open(
    "/Users/johnathanchiu/Downloads/san-jose-pd-firearm-sample.pdf"
) as pdf:
    crops = segment_pdf_page(pdf.pages[page_idx])

    im = pdf.pages[page_idx].to_image()
    for crop in crops:
        im.draw_rect(crop.bbox)

    im.show()


ifile = "/Users/johnathanchiu/Downloads/PM209/images/Apple_iphone-13-pro-max-07300325A-repair/images/Apple_iphone-13-pro-max-07300325A-repair_00014.jpg"
img = Image.open(ifile).convert("RGBA")

draw = ImageDraw.Draw(img)
for crop in segment_pdf_image(img):
    draw.rectangle(
        crop.bounding_box, outline=(255, 0, 0), width=3, fill=(0, 127, 255, 127)
    )

img.show()
