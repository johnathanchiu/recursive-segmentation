from PIL import Image, ImageDraw
import pdfplumber

from xyseg.document.segment import segment_pdf_page, segment_pdf_image

pdf_file = "examples/pdfs/san-jose-pd-firearm-sample.pdf"

page_idx = 0
with pdfplumber.open(pdf_file) as pdf:
    crops = segment_pdf_page(pdf.pages[page_idx])

    im = pdf.pages[page_idx].to_image()
    for crop in crops:
        im.draw_rect(crop.bounding_box)

    im.show()


ifile = "examples/images/apple_iphone-13_manual.jpg"
img = Image.open(ifile)

draw = ImageDraw.Draw(img, "RGBA")
for crop in segment_pdf_image(img):
    draw.rectangle(
        crop.bounding_box, outline=(255, 0, 0), width=3, fill=(0, 127, 255, 80)
    )

img.show()
